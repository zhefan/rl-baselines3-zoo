'''
Project: rl-baselines3-zoo
Created Date: Monday, March 29th 2021, 1:12:38 am
Author: Zhefan Ye <zhefanye@gmail.com>
-----
Copyright (c) 2021 TBD
Do whatever you want
'''

import os

import numpy as np
import gym


def check_path(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)


class RecordEnv(gym.Wrapper):
    """Record env experience"""

    def __init__(self, env, **kwargs):
        super().__init__(env)
        print('Using record environment')
        self.out_dir = kwargs['out_dir']
        self.info = kwargs['info']
        self.n_step = 0
        self.episode_num = -1  # starting at 0
        self.action_n = env.action_space.n
        self.env_id = env.spec.id

        self.s_rollout = []
        self.a_rollout = []
        self.r_rollout = []
        self.d_rollout = []

        check_path(os.path.join(self.out_dir, self.info))

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        action_vec = np.zeros(self.action_n)
        action_vec[action] = 1

        self.a_rollout += [action_vec]
        self.r_rollout += [reward]
        self.d_rollout += [done]
        self.s_rollout += [next_state]

        self.n_step += 1  # increment step
        if done:  # episode done -> save
            self._close()
        return next_state, reward, done, info

    def reset(self, **kwargs):
        self.n_step = 0
        self.episode_num += 1
        # clear all memory
        self.s_rollout.clear()
        self.a_rollout.clear()
        self.r_rollout.clear()
        self.d_rollout.clear()

        state = self.env.reset(**kwargs)
        self.s_rollout += [state]  # append the first state
        return state

    def _close(self):  # ! default close does not work
        # skip starting frames if space invaders
        trunc_idx = 125 if self.env_id == 'SpaceInvadersNoFrameskip-v4' else 0
        trunc_idx = 36 if self.env_id == 'SpaceInvaders-v4' else 0
        print(f"> End of rollout {self.episode_num}, {self.n_step} frames..."
              f"trunc idx: {trunc_idx}, saving {len(self.s_rollout[trunc_idx:])} frames")
        np.savez(os.path.join(self.out_dir, self.info, f"rollout_{self.episode_num}"),
                 observations=np.array(self.s_rollout[trunc_idx:]),
                 rewards=np.array(self.r_rollout[trunc_idx:]),
                 actions=np.array(self.a_rollout[trunc_idx:]),
                 terminals=np.array(self.d_rollout[trunc_idx:]))
