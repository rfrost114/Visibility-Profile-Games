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

class ActorNet(BaseNetwork):
    def __init__(
            self,
            num_nodes,
            num_defenders,
            feature_dim,
            pos_dim,
            hidden_dim,
            edge_index,
            lap_pos,
            walk_pos
    ):
        super().__init__(num_nodes, edge_index, lap_pos, walk_pos)
        self.num_defenders = num_defenders

        self.gnn = VirtualNodeNet(feature_dim, pos_dim, hidden_dim)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_defenders)
        )

        for module in self.actor_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.414)
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)

    def forward(self, observations):
        batch_size = observations.shape[0]
        x, lap, walk, edge, batch_idx = self._prepare_pyg_batch(observations)

        h, _ = self.gnn(x, lap, walk, edge, batch_idx)
        logits = self.actor_head(h) # logits shape [batch_size * num_nodes, num_defenders]

        # reshape to [batch, nodes, defenders]
        logits = logits.view(batch_size, self.num_nodes, self.num_defenders)

        # transpose nodes and defenders to match the action shape we specified
        logits.transpose(1,2) # [batch, defenders, nodes]

        # flatten 
        logits = logits.reshape(batch_size, -1)

        return logits