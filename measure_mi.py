'''
This script for Measuring Mutual Information in URL Benchmark using KNN method.

Methods:
- Uploaded snapshots form URL Benchmark repository.
- Approximate d(s|z) using KNN method for each skill z.
- Approximate d(s) using KNN method.
- Compute MI as  - E[log d(s)] + E[log d(s|z)].

Input: agent(skill_dim), domain, seed, snapshot_ts
Output: Estimated Mutual Information value.
'''

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
from pathlib import Path
import utils
import hydra
from dm_env import specs
import dmc
import utils
from tqdm import tqdm
from typing import Optional
from replay_buffer import ReplayBufferStorage, make_replay_loader

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

# ===============================================================  
# Workspace(For Given Agent(sf_dim) and Domain(obs, action_spec))
# ===============================================================
class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.domain, _ = self.cfg.task.split('_', 1)
        self.cfg.domain = self.domain

        # init env
        self.env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        
        # init agent
        self.agent = make_agent(cfg.obs_type,
                                self.env.observation_spec(),
                                self.env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        
        # initialize from pretrained
        if self.cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot(self.cfg.snapshot_ts)['agent']
            self.agent.init_from(pretrained_agent) 

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        self.sample_meta = []

        # init replay buffer for measuring mi
        data_specs = (self.env.observation_spec(),
                      self.env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,                 # not used
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None
        
        # save directory
        self.base_dir = Path("/scratch2/james2602/URLB/mi_measure_results")  # <- 네가 말한 base_dir (cfg에 없으면 cfg.snapshot_base_dir로 바꿔)
        self.save_dir = self.base_dir / str(self.cfg.agent.name) / str(self.cfg.domain) / str(self.cfg.seed) / str(self.cfg.snapshot_ts)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # init knn
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip=0.0, knn_k=6, knn_avg=True, knn_rms=True,
                             device=self.device, shift=1e-6)

        if self.domain in ['walker', 'quadruped']:
            self.total_collect_steps =self.agent.skill_dim * cfg.num_collect_episodes * 1000
        else:
            self.total_collect_steps =self.agent.skill_dim * cfg.num_collect_episodes * 250

        
        self.pbar = tqdm(
            total=self.total_collect_steps,
            desc="collecting samples",
            unit="step",
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.5,
            smoothing=0.1,
            disable=bool(int(os.environ.get('DISABLE_TQDM', '0'))) 
        )

        self.global_step = 0
        self.global_episode = 0

    
    # snapshot loading
    def load_snapshot(self, snapshot_ts):
            snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
            snapshot_dir = snapshot_base_dir / self.domain / self.cfg.agent.name
            def try_load(seed):
                snapshot = snapshot_dir / f'seed_{seed}' / 'snapshot' / f'snapshot_{snapshot_ts}.pt'
                if not snapshot.exists():
                    raise FileNotFoundError(
                        f"[load_snapshot] Snapshot not found.\n"
                        f"  expected: {snapshot}\n"
                        f"  domain={self.domain}, agent={self.cfg.agent.name}, seed={self.cfg.seed}, snapshot_ts={snapshot_ts}\n"
                        f"  hint: check cfg.snapshot_base_dir and directory structure."
                    )
                with snapshot.open('rb') as f:
                    payload = torch.load(f)
                return payload

            # try to load current seed
            payload = try_load(self.cfg.seed)
            if payload is not None:
                return payload
            else:
                raise FileNotFoundError(f"Snapshot not found for seed {self.cfg.seed} at {snapshot_dir}")
            

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    # measure mi
    def measure_mi(self):
        '''
        load snapshot -> collect samples -> measure mi
        '''
        
        collect_until_episode = utils.Until(self.cfg.num_collect_episodes,
                                            self.cfg.action_repeat)   
        
        # collect 2 episodes for each skill
        for z_index in range(self.agent.skill_dim):
            episode_step = 0
            self.global_episode = 0
            time_step = self.env.reset()
            meta = self.agent.init_distinct_meta(z_index)
            print(f"Collecting samples for skill {z_index}: {meta}")
            self.sample_meta.append(meta)
            self.replay_storage.add(time_step, meta)

            # collect samples               
            while collect_until_episode(self.global_episode):
                if time_step.last():
                    self.global_episode += 1
                    time_step = self.env.reset()
                    self.replay_storage.add(time_step, meta)
                    episode_step = 0

                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False)
                    
                # take env step
                time_step = self.env.step(action)
                self.replay_storage.add(time_step, meta)
                episode_step += 1
                self.global_step += 1

                self.pbar.update(1)
                self.pbar.set_postfix(ep=self.global_episode)
        self.pbar.close()
        # ====================================
        # E[log d(s)] 추정 (batch 1개)
        # ====================================
        batch = next(self.replay_iter)
        obs, action, reward, discount, next_obs, meta = utils.to_torch(batch, self.device)

        # obs -> representation (MIMAgent 스타일: encoder output이 s_dim)
        obs = self.agent.aug_and_encode(obs)  # [B, s_dim]

        with torch.no_grad():
            log_d_s_B = (-self.pbe(obs) * self.agent.skill_dim).squeeze(-1)  # [B]
            E_log_d_s = log_d_s_B.mean()                           # scalar

        # ===========================================
        # E[log d(s|z)] 추정 (z별로 batch 샘플링)
        # ===========================================
        E_log_d_s_given_z_list = []

        for z in range(self.agent.skill_dim):
            # 1) replay buffer에서 batch를 계속 뽑아서 meta가 z인 샘플만 모으기
            #    (최소 batch_size만큼 모일 때까지 반복)

            if self.cfg.agent.name in ['cic', 'mimdice']: # continuous skill
                target_skill = self.sample_meta[z]['skill']
            else: # one-hot skill
                target_skill = np.zeros(self.agent.skill_dim, dtype=np.float32)
                target_skill[z] = 1.0

            if self.cfg.agent.name in ['cic', 'diayn', 'cesd']:
                meta_spec_name = 'skill'
            elif self.cfg.agent.name in ['aps', 'mimdice']:
                meta_spec_name = 'task'
            elif self.cfg.agent.name in ['smm']:
                meta_spec_name = 'z'
            else:
                raise NotImplementedError(f"Unknown agent for meta spec name: {self.cfg.agent.name}")
            
            batch_np = self.replay_storage.sample_batch_by_meta(
                target_meta={meta_spec_name: target_skill},  
                batch_size=self.cfg.batch_size,
                atol=1e-6
            )
            print(f"Collected batch for skill {z}: {batch_np[6]}")

            obs_b, action_b, reward_b, discount_b, next_obs_b, meta_b = utils.to_torch(batch_np, self.device)
            rep_b = self.agent.aug_and_encode(obs_b)  # (B, rep_dim)
            print(f"rep_b: ${rep_b}")

            # 2) log d(s|z) 추정
            log_d_s_given_z = self.pbe(source=rep_b, target=rep_b)
            log_d_s_given_z = log_d_s_given_z.view(-1)                     # (B,)  

            # E[log d(s|z)]
            E_log_d_s_given_z = log_d_s_given_z.mean()
            E_log_d_s_given_z_list.append(E_log_d_s_given_z)  

        # z개 스칼라 -> 평균
        E_log_d_s_given_z = torch.stack(E_log_d_s_given_z_list).mean()

        mi_est = -E_log_d_s + E_log_d_s_given_z
        print(f"E[log d(s)]={E_log_d_s.item():.6f} | E[log d(s|z)]={E_log_d_s_given_z.item():.6f} | MI={mi_est.item():.6f}")
        
        # save results
        out_path = self.save_dir / f"mi_est.txt"
        out_path.write_text(f"{mi_est}\n")


@hydra.main(config_path='.', config_name='measure_mi')
def main(cfg):
    from measure_mi import Workspace as W
    root_dir = Path.cwd
    print("initializing workspace...")
    workspace = W(cfg)
    print("start measuring mi...")
    workspace.measure_mi() # agent, domain, seed, snapshot_ts -> outputs MI

if __name__ == '__main__':
    main()