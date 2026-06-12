import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class VirtualNodeNet(nn.Module):

    def __init__(self, feature_dim, pos_dim, hidden_dim):

        super(VirtualNodeNet, self).__init__()

        self.feature_layer = nn.Linear(feature_dim, hidden_dim)
        self.pos_layer = nn.Linear(pos_dim, hidden_dim)

        # virtual_node
        self.vn_embed = nn.Embedding(1, hidden_dim,)

        # for updating virtural node embedding after first message pass
        self.vn_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # for updating virtural node embedding after second message pass
        self.vn_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # graph attention layers
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, lap_pos, walk_pos, edge_index, batch_index):
        # inital feature embeddings 

        x_embed = self.feature_layer(x)
        pos_embed = self.pos_layer(torch.cat([lap_pos, walk_pos], dim=1))
        h = F.relu(x_embed + pos_embed)

        # create a virtual node for each node in the graph
        num_graphs = batch_index.max().item() + 1

        vn_h = self.vn_embed(torch.zeros(num_graphs, dtype=torch.long, device=x.device))

        # broadcast to physical nodes
        h = h + vn_h[batch_index]

        # message passing one
        h = self.gat1(h, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        # now update the virtual node based on the the message passing
        vn_update = global_mean_pool(h, batch_index)
        vn_h = vn_h + self.vn_mlp1(vn_update)
        vn_h = F.relu(vn_h)

        # same story for layer 2
        h = h + vn_h[batch_index]
        
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h)


        vn_update = global_mean_pool(h, batch_index)
        vn_h = vn_h + self.vn_mlp2(vn_update)
        vn_h = F.relu(vn_h)

        # h: node level embeds
        # vn_h : golbal node embedding

        return h, vn_h