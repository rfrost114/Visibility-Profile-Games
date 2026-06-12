from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class LoggingCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super().__init__(verbose)
        self.terminal_rewards = []
        self.captures = []
        self.final_rewards = []
        self.dense_rewards = []
    
    def _on_step(self):
        # log beta on each step
        for info in self.locals.get('infos', []):
            if 'captures' in info:
                self.logger.record('custom/beta', info['beta'])
                self.captures.append(info['captures'])
                self.terminal_rewards.append(info['terminal_reward'])
                self.final_rewards.append(info['final_reward'])
                self.dense_rewards.append(info['dense_rewards'])
        return True
    
    def _on_rollout_end(self):
        
        if len(self.captures):
            self.logger.record('custom/mean_captures', np.mean(self.captures))
            self.logger.record('custom/mean_terminal_rewards', np.mean(self.terminal_rewards))
            self.logger.record('custom/mean_final_rewards', np.mean(self.final_rewards))
            self.logger.record('custom/mean_dense_rewards', np.mean(self.dense_rewards))
            self.terminal_rewards = []
            self.captures = []
            self.final_rewards = []
            self.dense_rewards = []
        return True