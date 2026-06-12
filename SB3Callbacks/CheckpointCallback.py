import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path, iter_num, verbose=1):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
        self.save_path = save_path
        self.iter_num = iter_num
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])

            if (mean_reward > self.best_mean_reward) and self.num_timesteps > 4000:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                        print(f"\n[Model Saved] New best mean reward: {mean_reward:.3f}")
                path = os.path.join(self.save_path, f'best_model-iter-{self.iter_num}.zip')
                self.model.save(path)
                # f_path = os.path.join(self.save_path, 'best_model_performance.txt')
                # with open(f_path, 'w') as f:
                #     f.write(f'mean reward: {mean_reward:.3f} Time step: {self.num_timesteps}')
        return True