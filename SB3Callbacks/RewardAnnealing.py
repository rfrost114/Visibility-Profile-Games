from stable_baselines3.common.callbacks import BaseCallback

class RewardAnnealingCallback(BaseCallback):
    def __init__(self, inital_beta=0.99, verbose = 0):
        super(RewardAnnealingCallback, self).__init__(verbose)
        self.inital_beta = inital_beta

    def _on_step(self):
        # linear decay
        fraction_completed = self.num_timesteps / self.locals['total_timesteps']

        current_beta = self.inital_beta * (1 - fraction_completed)
        current_beta = max(0.0, current_beta)
        self.training_env.env_method('set_beta', current_beta)

        # exponential decay
        # current_beta = self.inital_beta ** (self.num_timesteps / 5)
        # current_beta = max(0.0, current_beta)
        # # print(self.num_timesteps)
        # self.training_env.env_method('set_beta', current_beta)
        # print(self.training_env.get_attr('beta'))

        return True