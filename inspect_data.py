'''
Project: rl-baselines3-zoo
Created Date: Monday, March 29th 2021, 4:43:59 pm
Author: Zhefan Ye <zhefanye@gmail.com>
-----
Copyright (c) 2021 TBD
Do whatever you want
'''

import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


def action_dict(action_vec, env):
    action = None
    if env == 'breakout':
        if action_vec[0] == 1:
            action = 'noop'
        elif action_vec[1] == 1:
            action = 'start'
        elif action_vec[2] == 1:
            action = 'right'
        elif action_vec[3] == 1:
            action = 'left'
        else:
            raise RuntimeError
    elif env == 'spaceinvaders':
        if action_vec[0] == 1:
            action = 'noop'
        elif action_vec[1] == 1:
            action = 'fire'
        elif action_vec[2] == 1:
            action = 'right'
        elif action_vec[3] == 1:
            action = 'left'
        elif action_vec[4] == 1:
            action = 'right & fire'
        elif action_vec[5] == 1:
            action = 'left & fire'
        else:
            raise RuntimeError
    else:
        action = str(np.nonzero(action_vec)[0].squeeze())
    return action


def inspect_experience(args):
    experience = np.load(args.rollout)
    with open(os.path.join(*(args.rollout.split('/')[:-2] + ['info.json']))) as f:
        info = json.load(f)
    env_name = info['env'].split('-')[0].replace('NoFrameskip', '').lower()
    print(len(experience['observations']))
    print(len(experience['actions']))
    print(len(experience['rewards']))
    print(len(experience['terminals']))
    h = experience['observations'][0].shape[0]
    w = experience['observations'][0].shape[1]

    plt.figure()
    display = plt.imshow(np.zeros((h, w, 3), dtype=np.uint8))
    for idx in range(len(experience['observations'])):
        state = experience['observations'][idx]
        action_vec = experience['actions'][idx]
        action = action_dict(action_vec, env_name)
        reward = experience['rewards'][idx]
        done = experience['terminals'][idx]
        print(f"step {idx}, action: {action}, reward: {reward}, done: {done}")
        display.set_data(state)
        plt.title(action)
        plt.pause(.05)
        # if action_vec[1] == 1:
        #     print(action)
            # breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout', type=str, required=True, help='rollout file path (.npz file)')
    args = parser.parse_args()
    inspect_experience(args)
