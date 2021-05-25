"""
Project: rl-baselines3-zoo
Created Date: Monday, May 24th 2021, 4:05:50 pm
Author: Zhefan Ye <zhefanye@gmail.com>
-----
Copyright (c) 2021 TBD
Do whatever you want
"""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class EpisodeRewCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(EpisodeRewCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        ep_info_buffer = self.model.ep_info_buffer
        rollout_ep_rew = self.model.rollout_buffer.rewards.sum(axis=0)
        if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
            avg_ep_rew = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
            avg_ep_len = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in ep_info_buffer]),
            )
            # TODO
            print(f"len(ep_info_buffer): {len(ep_info_buffer)}")
            print(f"avg_ep_rew: {avg_ep_rew}")
            print(f"avg_ep_len: {avg_ep_len}")
            print(
                f"len rollout_ep_rew: {self.model.rollout_buffer.rewards.shape} "
                f"sum rollout_ep_rew: {rollout_ep_rew}"
            )
            print("--------------------------------------")
