from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EarlyStoppingCallback(BaseCallback):

    def __init__(self, reward_thresh: float, verbose : int = 1):
        super().__init__(verbose)
        self.reward_thresh = reward_thresh

    def _on_step(self) -> bool:
        fraction_completed = self.num_timesteps / self.locals['total_timesteps']
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            # print([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            # mean reward over last 100 episodes

            if mean_reward >= self.reward_thresh and (fraction_completed > 0.05):
                if self.verbose > 0:
                    print(f'\n[Early Stop] Mean reward {mean_reward:.2f} crossed threshold {self.reward_thresh} {self.num_timesteps} steps')
                return False
        return True