import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from ActorNet import ActorNet
    from CriticNet import CriticNet
except ImportError:
    from networks.ActorNet import ActorNet
    from networks.CriticNet import CriticNet


class GNNPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        self.graph_kwargs = kwargs.pop("graph_kwargs")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
    
    def _build(self, lr_schedule):
        super()._build(lr_schedule)

        critic_kwargs = self.graph_kwargs.copy()
        critic_kwargs.pop('num_defenders')
        self.actor_net = ActorNet(**self.graph_kwargs)
        self.critic_net = CriticNet(**critic_kwargs)

        # ignore everything that sb3 usually does in here
        self.features_extractor = nn.Identity()
        self.mlp_extractor = nn.Identity()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def forward(self, obs, deterministic=False):
        action_logits = self.actor_net(obs)

        distribution = self.action_dist.proba_distribution(action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        values = self.critic_net(obs)

        return actions, values, log_prob
    
    def predict_values(self, obs):
        return self.critic_net(obs)

    def evaluate_actions(self, obs, actions):
        action_logits = self.actor_net(obs)
        distribution = self.action_dist.proba_distribution(action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic_net(obs)

        return values, log_prob, entropy
    
    def _predict(self, observation, deterministic = False):
        action_logits = self.actor_net(observation)
        distribution = self.action_dist.proba_distribution(action_logits)
        return distribution.get_actions(deterministic=deterministic)