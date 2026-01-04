import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, meta_specs, replay_dir):
        self._data_specs = data_specs
        self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, meta):
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            for spec in self._meta_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, storage, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        meta = []
        for spec in self._storage._meta_specs:
            meta.append(episode[spec.name][idx - 1])
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs, *meta)

    def __iter__(self):
        while True:
            yield self._sample()
    
    def _meta_match(self, meta_list, target_meta, atol=1e-6, rtol=0.0):
        """
        meta_list: _sample()에서 만드는 meta 리스트 (meta_specs 순서대로)
        target_meta: dict 또는 list/tuple
          - dict이면 {spec_name: value}
          - list/tuple이면 meta_specs 순서에 맞춘 값들
        """
        # meta_specs 순서
        meta_specs = self._storage._meta_specs

        # target_meta를 spec_name -> value dict로 통일
        if isinstance(target_meta, dict):
            target_dict = target_meta
        else:
            assert len(target_meta) == len(meta_specs), \
                f"target_meta length mismatch: {len(target_meta)} vs {len(meta_specs)}"
            target_dict = {spec.name: target_meta[i] for i, spec in enumerate(meta_specs)}

        # meta_list도 spec_name -> value로 매핑해서 비교
        for i, spec in enumerate(meta_specs):
            name = spec.name
            cur = meta_list[i]          # episode에서 뽑힌 meta 값 (np array)
            tgt = target_dict[name]     # 우리가 원하는 meta 값

            # 스칼라/배열 모두 np.array로 통일
            cur = np.asarray(cur)
            tgt = np.asarray(tgt)

            # dtype이 float면 allclose, 아니면 exact compare
            if np.issubdtype(cur.dtype, np.floating) or np.issubdtype(tgt.dtype, np.floating):
                if not np.allclose(cur, tgt, atol=atol, rtol=rtol):
                    return False
            else:
                if not np.array_equal(cur, tgt):
                    return False

        return True

    def sample_batch_by_meta(self, target_meta, batch_size, atol=1e-6, rtol=0.0, max_tries=200000):
        """
        target_meta에 해당하는 transition만 골라 batch_size개 모아서 반환.
        반환 형태: (obs_B, action_B, reward_B, discount_B, next_obs_B, *meta_B)
          - obs_B: (B, obs_dim) 등
          - meta_B는 meta spec마다 (B, meta_dim) 형태
        """
        # 최신 episode 파일을 가져오도록 fetch 시도
        self._try_fetch()

        obs_list, action_list, reward_list, discount_list, next_obs_list = [], [], [], [], []
        meta_lists = None  # meta spec 수만큼 list를 만들기 위해 나중에 초기화

        tries = 0
        while len(obs_list) < batch_size and tries < max_tries:
            tries += 1

            episode = self._sample_episode()
            idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

            # meta 추출 (idx-1에서)
            meta = []
            for spec in self._storage._meta_specs:
                meta.append(episode[spec.name][idx - 1])

            # meta가 target_meta와 일치하는지 검사
            if not self._meta_match(meta, target_meta, atol=atol, rtol=rtol):
                continue

            # 일치하면 transition 구성
            obs = episode['observation'][idx - 1]
            action = episode['action'][idx]
            next_obs = episode['observation'][idx + self._nstep - 1]

            reward = np.zeros_like(episode['reward'][idx])
            discount = np.ones_like(episode['discount'][idx])
            for i in range(self._nstep):
                step_reward = episode['reward'][idx + i]
                reward += discount * step_reward
                discount *= episode['discount'][idx + i] * self._discount

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            discount_list.append(discount)
            next_obs_list.append(next_obs)

            if meta_lists is None:
                meta_lists = [[] for _ in meta]
            for j in range(len(meta)):
                meta_lists[j].append(meta[j])

        if len(obs_list) < batch_size:
            raise RuntimeError(
                f"Could not collect enough samples for target_meta. "
                f"collected={len(obs_list)}/{batch_size}, tries={tries}/{max_tries}. "
                f"Try increasing max_tries or loosening atol."
            )

        # stack해서 batch 만들기
        obs_B = np.stack(obs_list, axis=0)
        action_B = np.stack(action_list, axis=0)
        reward_B = np.stack(reward_list, axis=0)
        discount_B = np.stack(discount_list, axis=0)
        next_obs_B = np.stack(next_obs_list, axis=0)

        meta_B = [np.stack(m, axis=0) for m in meta_lists]  # meta spec별로 (B, ...)

        return (obs_B, action_B, reward_B, discount_B, next_obs_B, *meta_B)


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(storage, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(storage,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader