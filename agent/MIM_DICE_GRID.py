# -*- coding: utf-8 -*-
"""
MIMAgent (continuous state) — batch-marginal ratio with Softplus DNet
--------------------------------------------------------------------
"""

import math
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent

import copy


# =============================================================================
# Utilities
# =============================================================================

def logmeanexp(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    """log(mean(exp(x))) = logsumexp(x) - log(n)"""
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.size(dim))


def get_f_divergence_fn(f_type: str, eps: float = 1e-12, exp_max: float = 60.0):
    """
    (f^*_+(y), f'^*_+^{-1}(y)) 반환.
    기본은 'softchi'(shifted-χ^2).
    """
    if f_type == 'softchi':
        def f_star(y):
            y_neg = y.clamp_max(0)
            return torch.where(y < 0, torch.expm1(y_neg), 0.5 * y * y + y)

        def f_prime_inv(y):
            y_neg = y.clamp_max(0)
            return torch.where(y < 0, torch.exp(y_neg), y + 1)

        return f_star, f_prime_inv

    elif f_type == 'kl':
        def f_star(y):
            y_clip = y.clamp(max=exp_max)
            return torch.expm1(y_clip)

        def f_prime_inv(y):
            return torch.exp((y - 1).clamp(max=exp_max))

        return f_star, f_prime_inv

    else:
        raise NotImplementedError(f'undefined f-divergence: {f_type}')


def schisq_log_relu_f_prime_inv(x):
    return torch.where(x < 0, x, torch.log(x.clamp_min(0) + 1))


def log_softplus_stable_masked(x: torch.Tensor,
                               neg_thr: float = -80.0,
                               pos_thr: float = 80.0) -> torch.Tensor:
    x32 = x.float()
    y64 = torch.empty_like(x32)

    neg = x32 <= neg_thr         # g(x) ≈ x
    pos = x32 >= pos_thr         # g(x) ≈ log(x)
    mid = ~(neg | pos)           # g(x) = log(log1p(exp(x)))

    y64[neg] = x32[neg]
    xp = x32[pos]
    y64[pos] = torch.log(xp)
    xm = x32[mid]
    y64[mid] = torch.log(F.softplus(xm))

    return y64.to(x.dtype)


# =============================================================================

class EqNet(nn.Module):
    """Predict e(s,a,z) for policy pretraining."""
    def __init__(self, in_dim, hidden_dim, num_layers: int = 2):
        super().__init__()
        layers = []
        if num_layers <= 0:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers += [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.net(x)


class LambdaNet(nn.Module):
    """λ(s) network."""
    def __init__(self, state_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.apply(utils.weight_init)

    def forward(self, s):
        return self.net(s)


class DNet(nn.Module):
    def __init__(self, obs_task_dim, hidden, K, num_layers=2):
        super().__init__()
        feats = []
        if num_layers <= 0:
            feats.append(nn.Linear(obs_task_dim, 1))
        else:
            feats += [nn.Linear(obs_task_dim, hidden), nn.LeakyReLU(inplace=True)]
            for _ in range(num_layers - 1):
                feats += [nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True)]
            feats.append(nn.Linear(hidden, 1))
        self.core = nn.Sequential(*feats)
        self.act = nn.Softplus()
        self.task_num = K
        self.apply(utils.weight_init)

    def forward(self, s_task, return_preact=False, return_both=False):
        a = self.core(s_task)
        if return_both:
            return a, self.act(a)
        return a if return_preact else self.act(a)


class PhiNet(nn.Module):
    """
    φ(z) 네트워크: task one-hot z ∈ R^K → scalar φ_z.
    θ-step(최소화)에서 업데이트되고, q-step(최대화)에서는 detach되어 고정됩니다.
    """
    def __init__(self, K: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, z_onehot: torch.Tensor) -> torch.Tensor:
        # input: [..., K], output: [..., 1]
        return self.net(z_onehot)


class VNet(nn.Module):
    """v(s,z) network."""
    def __init__(self, obs_task_dim: int, hidden: int, K: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_task_dim, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.apply(utils.weight_init)
        self.task_num = K

    def forward(self, s_task):
        return self.net(s_task)


# =============================================================================
# MIM Agent (continuous state) — batch-marginal with Softplus DNet
# =============================================================================
class MIMAgent(DDPGAgent):
    """
    MIM Agent (continuous state) — batch-marginal with Softplus QNet
    - log r(s,z) = log q(s|z) - log p_marg(s)
    - θ-step: minimize wrt {λ, μ, v, φ(z)} (q detached)
    - q-step: maximize wrt q(s|z) (θ detached), with IS-normalization via φ
    """
    def __init__(self,
                 update_task_every_step: int = 100,
                 sf_dim: int = 16,
                 skill_dim: int =16,
                 alpha: float = 1.0,
                 gamma: float = 0.99,
                 d_hidden: int = 256,
                 num_d_layer: int = 2,
                 d_s_z_lr: float = 0.1,   # DNet lr ratio (w.r.t. base lr)
                 phi_lr: float = 0.01,    # PhiNet lr ratio (w.r.t. base lr)
                 lstsq_batch_size: int = 2048,
                 update_encoder: bool = False,
                 # ---- [추가] meta regression 관련 하이퍼파라미터 ----
                 num_init_steps: int = 25000,     # 2만5천 step 전까지 랜덤 task
                 num_second_steps: int = 40000, 
                 meta_opt_lr: float = 1e-4,
                 meta_opt_iters: int = 10000,
                 **kwargs):

        # DDPGAgent에 task(meta) 차원을 알려줌
        kwargs["meta_dim"] = sf_dim
        super().__init__(**kwargs)

        # 기본 차원 설정
        self.sf_dim = int(sf_dim)
        self.skill_dim = int(sf_dim) # Since hyperparmeter
        self.s_dim = self.obs_dim - self.sf_dim  # obs = [state, task_onehot]

        # Hyperparams
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.d_s_z_lr = float(d_s_z_lr)
        self.phi_lr = float(phi_lr)
        self.d_hidden = int(d_hidden)
        self.num_d_layer = int(num_d_layer)

        # θ = λ, μ, v
        self.lam_net = LambdaNet(self.s_dim, self.hidden_dim).to(self.device)
        self.v_net = VNet(self.obs_dim, self.hidden_dim, self.sf_dim).to(self.device)
        # em_opt: λ, μ, v 만 포함 (φ 제외)
        self.em_opt = torch.optim.Adam(
            itertools.chain(self.lam_net.parameters(),
                            self.v_net.parameters()),
            lr=self.lr
        )

        # φ(z)만 별도 옵티마이저 및 lr ratio
        self.phi_net = PhiNet(self.sf_dim, self.hidden_dim).to(self.device)
        self.phi_opt = torch.optim.Adam(self.phi_net.parameters(), lr=self.lr * self.phi_lr)

        # Policy pretraining auxiliary (e-predictor)
        self.eq_net = EqNet(self.obs_dim + self.action_dim, self.d_hidden,
                            num_layers=self.num_d_layer).to(self.device)
        self.eq_opt = torch.optim.Adam(self.eq_net.parameters(), lr=self.lr)

        # Critic

        # D(s|z) with its own lr ratio
        self.d_net = DNet(self.obs_dim, self.d_hidden, self.sf_dim,
                          num_layers=self.num_d_layer).to(self.device)
        self.d_opt = torch.optim.Adam(self.d_net.parameters(), lr=self.lr * self.d_s_z_lr)

        self.train()
        self.critic_target.train()

        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip=0.0, knn_k=6, knn_avg=True, knn_rms=True,
                             device=self.device, shift=1e-6)

        self.update_task_every_step = update_task_every_step

        self._q_eps = 1e-5000

        self.ft_returns = np.zeros(self.sf_dim, dtype=np.float32)
        self.ft_not_finished = [True for _ in range(self.sf_dim)]
        self.pick_all_round = 3
        # self.META = None
        self.used_skills = np.zeros(self.sf_dim, dtype=np.float32) # not used

        # ---- [추가] meta selection / regression 상태 ----
        self.num_init_steps = int(num_init_steps)
        self.num_second_steps = int(num_second_steps)
        self.meta_opt_lr = float(meta_opt_lr)
        self.meta_opt_iters = int(meta_opt_iters)
        self.solved_meta = None  # regress_meta가 찾은 최적 task(meta)
        self.solved_second_meta = None

    # -------------------------------------------------------------------------
    # Meta / Task API
    # -------------------------------------------------------------------------
    def get_meta_specs(self):
        return (specs.Array((self.sf_dim,), np.float32, 'task'),)

    def _sample_random_task(self):
        """정규화 없는 유니폼 task 벡터 (reward-free와 동일)."""
        task = np.random.uniform(0.0, 1.0, self.sf_dim).astype(np.float32)
        return task

    def init_from(self, other_agent):
        """
        Pretrain MIMAgent에서 학습된 모든 네트워크 파라미터를 복사.
        - encoder / actor / critic / critic_target (부모에서 하던 것)
        - lam_net, v_net, d_net, phi_net, eq_net 까지 포함
        """
        super().init_from(other_agent)

        # 2) MIM 전용 네트워크들 복사
        self.lam_net.load_state_dict(other_agent.lam_net.state_dict())
        self.v_net.load_state_dict(other_agent.v_net.state_dict())
        self.d_net.load_state_dict(other_agent.d_net.state_dict())
        self.phi_net.load_state_dict(other_agent.phi_net.state_dict())
        self.eq_net.load_state_dict(other_agent.eq_net.state_dict())

    def init_meta(self):
        # reward_free면 기존처럼 랜덤 task 사용
        if self.reward_free:
            task = np.random.uniform(0, 1, self.sf_dim).astype(np.float32)
            return OrderedDict(task=task)

        if self.solved_second_meta is not None:
            task = np.array(self.solved_second_meta['task'], copy=True)
            return OrderedDict(task=task.astype(np.float32))
        
        if self.solved_meta is not None:
            task = np.array(self.solved_meta['task'], copy=True)
            return OrderedDict(task=task.astype(np.float32))
        task = self._sample_random_task()
        return OrderedDict(task=task)
    
    def init_distinct_meta(self, z_index):
        # reward_free면 기존처럼 랜덤 task 사용
        if self.reward_free:
            task = np.random.uniform(0, 1, self.sf_dim).astype(np.float32)
            return OrderedDict(task=task)

        if self.solved_second_meta is not None:
            task = np.array(self.solved_second_meta['task'], copy=True)
            return OrderedDict(task=task.astype(np.float32))
        
        if self.solved_meta is not None:
            task = np.array(self.solved_meta['task'], copy=True)
            return OrderedDict(task=task.astype(np.float32))
        task = self._sample_random_task()
        return OrderedDict(task=task)

    def update_meta(self, meta, global_step, time_step):
        if self.reward_free:
            if global_step % self.update_task_every_step == 0:
                return self.init_meta()
            return meta
        
        if self.solved_second_meta is not None:
            task = np.array(self.solved_second_meta['task'], copy=True)
            return OrderedDict(task=task.astype(np.float32))
        
        if self.solved_meta is not None:
            meta['task'] = np.array(self.solved_meta['task'], copy=True)
            return meta
        return meta

    def regress_meta(self, replay_iter, step):

        if self.solved_second_meta is not None:
            return self.solved_second_meta
        
        if step > self.num_second_steps:
            device = self.device

            # 1) z 초기화 (연속 벡터, [0,1] 범위)
            z0 = self._sample_random_task()                       # np array in [0,1]
            z0_clipped = np.clip(z0, 1e-4, 1.0 - 1e-4)            # logit 발산 방지
            z0_logit = np.log(z0_clipped) - np.log(1.0 - z0_clipped)  # inverse sigmoid

            # z_raw는 제한 없는 파라미터, 실제 z = sigmoid(z_raw)
            z_raw = torch.from_numpy(z0_logit).to(device).float()
            z_raw = z_raw.clone().detach().requires_grad_(True)

            # scale / shift 파라미터 (a>0, b 자유)
            a_raw = torch.zeros(1, device=device, requires_grad=True)  # softplus로 양수 강제
            b = torch.zeros(1, device=device, requires_grad=True)

            # lam_net, d_net은 고정된 feature extractor로 사용 (grad X)
            for p in self.d_net.parameters():
                p.requires_grad_(False)
            for p in self.lam_net.parameters():
                p.requires_grad_(False)

            optimizer = torch.optim.Adam([z_raw, a_raw, b], lr=self.meta_opt_lr)

            # 2) meta_opt_iters 만큼 SGD
            for _ in range(self.meta_opt_iters):
                batch = next(replay_iter)
                batch_obs, _, batch_reward, *_ = utils.to_torch(batch, device)

                with torch.no_grad():
                    s_enc = self.aug_and_encode(batch_obs)   # [B, s_dim]
                    lam_s = self.lam_net(s_enc)              # [B, 1]

                B = s_enc.size(0)
                optimizer.zero_grad(set_to_none=True)

                # 실제로 사용하는 z는 sigmoid(z_raw) → 항상 0~1
                z = torch.sigmoid(z_raw)                     # [sf_dim]
                z_exp = z.unsqueeze(0).expand(B, -1)         # [B, sf_dim]
                sz = torch.cat([s_enc, z_exp], dim=-1)       # [B, obs_dim]

                a_pre, _ = self.d_net(sz, return_both=True)  # [B,1]
                log_d = log_softplus_stable_masked(
                    a_pre, neg_thr=-80.0, pos_thr=80.0
                )                                           # [B,1]

                pred_raw = lam_s + log_d                    # [B,1]

                # a>0 보장
                a = F.softplus(a_raw) + 1e-6
                pred = a * pred_raw + b                     # [B,1]

                loss = F.mse_loss(pred, batch_reward)
                loss.backward()
                optimizer.step()

            # 3) lam_net, d_net 다시 gradient 켜주기
            for p in self.d_net.parameters():
                p.requires_grad_(True)
            for p in self.lam_net.parameters():
                p.requires_grad_(True)

            # 4) 최종 z = sigmoid(z_raw)를 meta로 저장 (항상 [0,1] 범위)
            with torch.no_grad():
                z_final = torch.sigmoid(z_raw)
                task_np = z_final.detach().cpu().numpy().astype(np.float32)
                a_val = (F.softplus(a_raw) + 1e-6).item()
                b_val = b.item()

            meta = OrderedDict()
            meta['task'] = task_np
            self.solved_second_meta = meta

            print(
                f"[MIMAgent] solved_second_meta fixed at step {step}: "
                f"task={meta['task']}, scale={a_val:.4f}, shift={b_val:.4f}"
            )
            return meta


        
        if self.solved_meta is not None:
            return self.solved_meta

        # 충분히 rollout 하기 전에는 그냥 랜덤 task 사용
        if step < self.num_init_steps:
            meta = OrderedDict()
            meta['task'] = self._sample_random_task()
            return meta

        device = self.device

        # 1) z 초기화 (연속 벡터, [0,1] 범위)
        z0 = self._sample_random_task()                       # np array in [0,1]
        z0_clipped = np.clip(z0, 1e-4, 1.0 - 1e-4)            # logit 발산 방지
        z0_logit = np.log(z0_clipped) - np.log(1.0 - z0_clipped)  # inverse sigmoid

        # z_raw는 제한 없는 파라미터, 실제 z = sigmoid(z_raw)
        z_raw = torch.from_numpy(z0_logit).to(device).float()
        z_raw = z_raw.clone().detach().requires_grad_(True)

        # scale / shift 파라미터 (a>0, b 자유)
        a_raw = torch.zeros(1, device=device, requires_grad=True)  # softplus로 양수 강제
        b = torch.zeros(1, device=device, requires_grad=True)

        # lam_net, d_net은 고정된 feature extractor로 사용 (grad X)
        for p in self.d_net.parameters():
            p.requires_grad_(False)
        for p in self.lam_net.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.Adam([z_raw, a_raw, b], lr=self.meta_opt_lr)

        # 2) meta_opt_iters 만큼 SGD
        for _ in range(self.meta_opt_iters):
            batch = next(replay_iter)
            batch_obs, _, batch_reward, *_ = utils.to_torch(batch, device)

            with torch.no_grad():
                s_enc = self.aug_and_encode(batch_obs)   # [B, s_dim]
                lam_s = self.lam_net(s_enc)              # [B, 1]

            B = s_enc.size(0)
            optimizer.zero_grad(set_to_none=True)

            # 실제로 사용하는 z는 sigmoid(z_raw) → 항상 0~1
            z = torch.sigmoid(z_raw)                     # [sf_dim]
            z_exp = z.unsqueeze(0).expand(B, -1)         # [B, sf_dim]
            sz = torch.cat([s_enc, z_exp], dim=-1)       # [B, obs_dim]

            a_pre, _ = self.d_net(sz, return_both=True)  # [B,1]
            log_d = log_softplus_stable_masked(
                a_pre, neg_thr=-80.0, pos_thr=80.0
            )                                           # [B,1]

            pred_raw = lam_s + log_d                    # [B,1]

            # a>0 보장
            a = F.softplus(a_raw) + 1e-6
            pred = a * pred_raw + b                     # [B,1]

            loss = F.mse_loss(pred, batch_reward)
            loss.backward()
            optimizer.step()

        # 3) lam_net, d_net 다시 gradient 켜주기
        for p in self.d_net.parameters():
            p.requires_grad_(True)
        for p in self.lam_net.parameters():
            p.requires_grad_(True)

        # 4) 최종 z = sigmoid(z_raw)를 meta로 저장 (항상 [0,1] 범위)
        with torch.no_grad():
            z_final = torch.sigmoid(z_raw)
            task_np = z_final.detach().cpu().numpy().astype(np.float32)
            a_val = (F.softplus(a_raw) + 1e-6).item()
            b_val = b.item()

        meta = OrderedDict()
        meta['task'] = task_np
        self.solved_meta = meta

        print(
            f"[MIMAgent] solved_meta fixed at step {step}: "
            f"task={meta['task']}, scale={a_val:.4f}, shift={b_val:.4f}"
        )

        return meta



    # -------------------------------------------------------------------------
    # Main update
    # -------------------------------------------------------------------------
    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        # --- Sample batch ---
        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, task = utils.to_torch(batch, self.device)

        # --- Encode (and possibly augment) ---
        obs = self.aug_and_encode(obs)       # [B, s_dim]
        next_obs = self.aug_and_encode(next_obs)  # [B, s_dim]

        # -------------------- Reward-free pretraining --------------------
        if self.reward_free:
            device, dtype = obs.device, obs.dtype
            B = obs.size(0)
            K = min(4, int(self.sf_dim))

            # Expand with K one-hot tasks
            tasks_K = torch.empty(K, self.sf_dim, device=device, dtype=dtype).uniform_(0.0, 1.0)
            task_BK = tasks_K.unsqueeze(0).expand(B, K, -1)                       # [B,K,Z]
            s_BK = obs.unsqueeze(1).expand(B, K, -1)                              # [B,K,s_dim]
            sN_BK = next_obs.unsqueeze(1).expand(B, K, -1)                        # [B,K,s_dim]
            pairs_BK = torch.cat([s_BK, task_BK], dim=-1).reshape(B * K, -1)      # [B*K, obs_dim]
            pairs_nextBK = torch.cat([sN_BK, task_BK], dim=-1).reshape(B * K, -1) # [B*K, obs_dim]
            s_flat = s_BK.reshape(B * K, -1)
            z_flat = task_BK.reshape(B * K, -1)
            act_BK = action.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

            # --- EM-like saddle step (θ-descend, q-ascend) ---
            m_em = self.update_Mstep(obs, B, K, pairs_BK, pairs_nextBK, tasks_K)
            metrics.update({f"em/{k}": v for k, v in m_em.items()})

            # --- Intrinsic reward for critic pretrain ---
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, pairs_BK, pairs_nextBK, B, K, step)
                discount_BK = discount.unsqueeze(1).expand(B, K, -1).reshape(B * K, 1)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward_mean'] = float(intr_reward.mean().item())
                metrics['intr_reward_std'] = float(intr_reward.std(unbiased=False).item())

            # --- Policy pretraining ---
            metrics.update({f"policy/{k}": v for k, v in
                            self.update_actor_pre(obs, act_BK.detach(), step, B, K,
                                                  pairs_BK, pairs_nextBK).items()})

            # --- Critic pretraining ---
            metrics.update({f"policy/{k}": v for k, v in
                            self.update_critic(pairs_BK, act_BK.detach(), intr_reward,
                                               discount_BK, pairs_nextBK, step).items()})

            # --- Target update ---
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            return metrics

        # -------------------- Fine-tuning (extrinsic) --------------------
        reward = extr_reward
        obs_cat = torch.cat([obs, task], dim=1)
        next_cat = torch.cat([next_obs, task], dim=1)
        metrics.update(self.update_critic(obs_cat.detach(), action, reward,
                                          discount, next_cat.detach(), step))
        metrics.update(self.update_actor(obs_cat.detach(), step))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = float(extr_reward.mean().item())
            metrics['batch_reward'] = float(reward.mean().item())
        return metrics

    # -------------------------------------------------------------------------
    # MIM-DICE saddle step (continuous) — θ: descend, q: ascend
    # -------------------------------------------------------------------------
    def update_Mstep(self, obs, B, K, pairs_BK, pairs_nextBK, tasks_K):
        """
        obs: [B, s_dim]
        pairs_*: [B*K, obs_dim]  (obs||task)
        """
        metrics = {}
        f_star, _ = get_f_divergence_fn('softchi')

        # ---------- 공통 값 ----------
        lam_B1 = self.lam_net(obs).reshape(B, 1)                  # [B,1]
        v_BK = self.v_net(pairs_BK).reshape(B, K)                 # [B,K]
        vN_BK = self.v_net(pairs_nextBK).reshape(B, K)            # [B,K]

        a_BK, d_BK = self.d_net(pairs_BK, return_both=True)
        a_BK = a_BK.reshape(B, K)
        d_BK = d_BK.reshape(B, K)

        log_d_BK = log_softplus_stable_masked(a_BK, neg_thr=-80.0, pos_thr=80.0)

        with torch.no_grad():
            log_marg_B = (- self.pbe(obs) * self.s_dim).squeeze(-1)  # [B]
            p_marg_B = log_marg_B.exp().clamp_min(1e-12)             # [B]

        # ---------------- θ-step (minimize wrt {λ, μ, v, φ}; q detached) ----------------
        # L1 ≈ logmeanexp_s [-λ(s) - log p_marg(s)]
        term1_theta = logmeanexp(-lam_B1.squeeze(-1) - log_marg_B, dim=0)

        # Lf (f*-항): e = log q (detached in θ-step) + γ v' - v + μ
        e_BK_theta = self.gamma * vN_BK - v_BK + lam_B1 + log_d_BK.detach()
        term2_theta = self.alpha * f_star(e_BK_theta / self.alpha).mean()

        # Ls
        term3_theta = (1.0 - self.gamma) * v_BK.mean()

        # Lphi (IS): E_{p_marg}[ q/p_marg ] - 1, φ(z) from deep net
        phi_K = self.phi_net(tasks_K).squeeze(-1)                          # [K]
        w_norm_theta = (d_BK.detach() / p_marg_B[:, None])                 # [B,K]
        term4_theta = torch.dot(w_norm_theta.mean(dim=0) - 1.0, phi_K)

        loss_theta = term1_theta + term2_theta + term3_theta + term4_theta

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.em_opt.zero_grad(set_to_none=True)   # λ, μ, v
        self.phi_opt.zero_grad(set_to_none=True)  # φ only
        loss_theta.backward()
        self.em_opt.step()
        self.phi_opt.step()
        if self.encoder_opt is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=100.0)
            self.encoder_opt.step()

        # ---------------- q-step (maximize wrt q; θ, φ detached) ----------------
        with torch.no_grad():
            lam_B1_det = lam_B1.detach()
            v_BK_det = v_BK.detach()
            vN_BK_det = vN_BK.detach()
            phi_K_det = self.phi_net(tasks_K).squeeze(-1).detach()

        # Lf with q active
        e_BK_phi = self.gamma * vN_BK_det - v_BK_det + lam_B1_det + log_d_BK
        term2_phi = self.alpha * f_star(e_BK_phi / self.alpha).mean()

        # Lphi (IS): φ detached, q만 업데이트
        w_norm_phi = (d_BK / p_marg_B[:, None])                          # [B,K]
        term4_phi = torch.dot(w_norm_phi.mean(dim=0) - 1.0, phi_K_det)

        loss_phi = term2_phi + term4_phi

        self.d_opt.zero_grad(set_to_none=True)
        (-loss_phi).backward()   # maximize wrt q_net parameters
        self.d_opt.step()

        metrics.update({
            'loss_theta': float(loss_theta.item()),
            'loss_phi': float(loss_phi.item()),
            'term1': float(term1_theta.item()),
            'term2': float(term2_theta.item()),
            'term3': float(term3_theta.item()),
            'term4': float(term4_theta.item()),
        })
        return metrics

    # -------------------------------------------------------------------------
    # Intrinsic reward (log r + μ * g(e/α))
    # -------------------------------------------------------------------------
    def compute_intr_reward(self, obs, pairs_BK, pairs_nextBK, B, K, step):
        with torch.no_grad():
            lam_B1 = self.lam_net(obs).reshape(B, 1)

            a_BK, _ = self.d_net(pairs_BK, return_both=True)
            a_BK = a_BK.reshape(B, K)
            log_d_BK = log_softplus_stable_masked(a_BK, neg_thr=-80.0, pos_thr=80.0)


            rep = lam_B1 + log_d_BK 
        return rep.reshape(B * K, 1)

    # -------------------------------------------------------------------------
    # Policy pretraining (uses e-predictor and g(e/α) weights)
    # -------------------------------------------------------------------------
    def update_actor_pre(self, obs, act_BK, step, B, K, pairs_BK, pairs_nextBK):
        metrics = {}
        with torch.no_grad():
            lam_B1 = self.lam_net(obs).reshape(B, 1)
            v_BK = self.v_net(pairs_BK).reshape(B, K)
            vN_BK = self.v_net(pairs_nextBK).reshape(B, K)
            a_BK, _ = self.d_net(pairs_BK, return_both=True)
            a_BK = a_BK.reshape(B, K)
            with torch.cuda.amp.autocast(enabled=False):
                log_d_BK = log_softplus_stable_masked(a_BK, neg_thr=-80.0, pos_thr=80.0)
            e_BK = log_d_BK + self.gamma * vN_BK - v_BK + lam_B1
            target_e = e_BK.reshape(B * K, 1)                      # [B*K,1]

        # ----- EqNet 학습 (데이터 행동) -----
        eq_in_data = torch.cat([pairs_BK, act_BK], dim=-1)
        pred_e_data = self.eq_net(eq_in_data)
        eq_loss = F.mse_loss(pred_e_data, target_e)
        self.eq_opt.zero_grad(set_to_none=True)
        eq_loss.backward()
        self.eq_opt.step()

        # ----- 정책 스텝 (가중치 g(e/α)) -----
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(pairs_BK, stddev)
        act_pi = dist.sample(clip=self.stddev_clip)

        eq_in_pi = torch.cat([pairs_BK, act_pi], dim=-1)
        e_pred_pi = self.eq_net(eq_in_pi)                         # [B*K,1]
        log_w = schisq_log_relu_f_prime_inv(e_pred_pi / self.alpha)

        policy_loss = -(log_w.mean())
        self.actor_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics["eq_loss"] = float(eq_loss.item())
            metrics["eq_pred_mean"] = float(pred_e_data.mean().item())
            metrics['actor_loss'] = float(policy_loss.item())
            metrics['log_w'] = float(log_w.mean().item())
        return metrics

    # -------------------------------------------------------------------------
    # Fine-tuning (extrinsic) – 예전 보조 함수들 (사용 안 해도 됨)
    # -------------------------------------------------------------------------
    def update_actor_finetune(self, obs, task, step):
        metrics = {}
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        obs = torch.cat([obs, task], dim=1)
        Q1, Q2 = self.critic(obs, action)
        actor_loss = -torch.min(Q1, Q2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = float(actor_loss.item())
            metrics['actor_logprob'] = float(log_prob.mean().item())
            metrics['actor_ent'] = float(dist.entropy().sum(dim=-1).mean().item())
        return metrics

    def update_critic_finetune(self, obs, action, reward, discount, next_obs, task, step):
        metrics = {}
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, task)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = float(target_Q.mean().item())
            metrics['critic_q1'] = float(Q1.mean().item())
            metrics['critic_q2'] = float(Q2.mean().item())
            metrics['critic_loss'] = float(critic_loss.item())
        return metrics
