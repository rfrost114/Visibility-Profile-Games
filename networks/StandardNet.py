import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class StandardNet(nn.Module):

    def __init__(self, feature_dim, pos_dim, hidden_dim):

        super(StandardNet, self).__init__()

        self.feature_layer = nn.Linear(feature_dim, hidden_dim)
        self.pos_layer = nn.Linear(pos_dim, hidden_dim)


        # graph attention layers
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

    def forward(self, x, lap_pos, walk_pos, edge_index, batch_index):
        # inital feature embeddings 

        x_embed = self.feature_layer(x)
        pos_embed = self.pos_layer(torch.cat([lap_pos, walk_pos], dim=1))
        h = F.relu(x_embed + pos_embed)

        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = F.elu(h)

        state_embed = global_mean_pool(h, batch_index)

        return h, state_embed