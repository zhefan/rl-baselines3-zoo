'''
Project: rl-baselines3-zoo
Created Date: Monday, March 29th 2021, 4:43:59 pm
Author: Zhefan Ye <zhefanye@gmail.com>
-----
Copyright (c) 2021 TBD
Do whatever you want
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt


def action_dict(action_vec, game):
    action = None
    if game == 'breakout':
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
    elif game == 'spaceinvaders':
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
        raise NotImplementedError
    return action


def inspect_experience(args):
    experience = np.load('datasets/BreakoutNoFrameskip-v4/rollout_0.npz') \
        if args.game == 'breakout' \
        else np.load('datasets/SpaceInvadersNoFrameskip-v4/rollout_0.npz')
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
        action = action_dict(action_vec, args.game)
        reward = experience['rewards'][idx]
        done = experience['terminals'][idx]
        print(f"step {idx}, action: {action}, reward: {reward}, done: {done}")
        display.set_data(state)
        plt.title(action)
        plt.pause(.05)
        if action_vec[2] == 1:
            print(action)
            # breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='spaceinvaders',
                        help='spaceinvaders | breakout')
    args = parser.parse_args()
    inspect_experience(args)
