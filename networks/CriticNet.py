import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from BaseNetwork import BaseNetwork
    from VirtualNodeNet import VirtualNodeNet
    from StandardNet import StandardNet
except ImportError:
    from networks.BaseNetwork import BaseNetwork
    from networks.VirtualNodeNet import VirtualNodeNet
    from networks.StandardNet import StandardNet

class CriticNet(BaseNetwork):
    def __init__(
            self,
            num_nodes,
            num_defenders,
            num_attacker_policies,
            feature_dim,
            pos_dim,
            hidden_dim,
            edge_index,
            lap_pos,
            walk_pos
    ):
        super().__init__(num_nodes, edge_index, lap_pos, walk_pos)
        self.gnn = VirtualNodeNet(feature_dim + num_defenders, pos_dim, hidden_dim)
        self.attacker_policy_embed = nn.Embedding(num_attacker_policies, hidden_dim)

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs_dict):

        public_obs = obs_dict['public_obs']
        attacker_policy_id = obs_dict['private_obs'].long().view(-1)

        x, lap, walk, edge, batch_idx = self._prepare_pyg_batch(public_obs)

        _, vn_h = self.gnn(x, lap, walk, edge, batch_idx)

        attacker_context = self.attacker_policy_embed(attacker_policy_id)

        policy_informed_vn_h = torch.cat([vn_h, attacker_context], dim=1)

        state_value = self.critic_head(policy_informed_vn_h).squeeze(-1) # [batch_size]
        return state_value