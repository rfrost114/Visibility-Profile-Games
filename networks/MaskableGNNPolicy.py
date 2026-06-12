from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch.nn as nn
try:
    from AutoRegressiveActorNet import AutoRegressiveActorNet
    from CriticNet import CriticNet
except ImportError:
    from networks.AutoRegressiveActorNet import AutoRegressiveActorNet
    from networks.CriticNet import CriticNet

class MaskableGNNPolicy(MaskableActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        self.graph_kwargs = kwargs.pop('graph_kwargs')
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
    
    def _build(self, lr_schedule):
        super()._build(lr_schedule)

        actor_kwargs = self.graph_kwargs.copy()
        actor_kwargs.pop('num_attacker_policies')

        self.actor_net = AutoRegressiveActorNet(**actor_kwargs)
        self.critic_net = CriticNet(**self.graph_kwargs)
        self.features_extractor = nn.Identity()
        self.mlp_extractor = nn.Identity()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def forward(self, obs_dict, deterministic = False, action_masks = None):
        action_logits = self.actor_net(obs_dict['public_obs'])

        # pass action mask to prob dist as well
        distribution = self.action_dist.proba_distribution(action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.critic_net(obs_dict)

        return actions, values, log_prob
    
    def evaluate_actions(self, obs_dict, actions, action_masks = None):
        action_logits = self.actor_net(obs_dict['public_obs'])
        distribution = self.action_dist.proba_distribution(action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic_net(obs_dict)

        return values, log_prob, entropy
    
    def predict_values(self, obs_dict):
        return self.critic_net(obs_dict)
    
    def _predict(self, obs_dict, deterministic = False, action_masks = None):
        action_logits = self.actor_net(obs_dict['public_obs'])
        distribution = self.action_dist.proba_distribution(action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution.get_actions(deterministic=deterministic)
