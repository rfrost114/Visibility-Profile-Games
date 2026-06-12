import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(
            self,
            num_nodes,
            edge_index,
            lap_pos,
            walk_pos,
    ):
        super(BaseNetwork, self).__init__()
        
        self.num_nodes = num_nodes
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('lap_pos', lap_pos)
        self.register_buffer('walk_pos', walk_pos)
    
    # observations are in a [batch_size, num_nodes, feature_dim] tensor, pyG requires a slightly different format because it hates me
    def _prepare_pyg_batch(self, observations : torch.Tensor):
        batch_size = observations.shape[0]
        device = observations.device

        # flatten into pyG format
        x = observations.view(batch_size * self.num_nodes, -1)

        # batch assignment array: maps each node to the graph its in 
        batch_index = torch.arange(batch_size, device=device).repeat_interleave(self.num_nodes)

        # edge indicies need to be tiled and relabled 
        num_edges = self.edge_index.shape[1]
        batched_edge_index = self.edge_index.repeat(1, batch_size)
        offsets = torch.arange(batch_size, device=device) * self.num_nodes
        offsets = offsets.repeat_interleave(num_edges)
        batched_edge_index = batched_edge_index + offsets

        # tile positional encodings
        batched_lap_pos = self.lap_pos.repeat(batch_size, 1)
        batched_walk_pos = self.walk_pos.repeat(batch_size, 1)

        return x, batched_lap_pos, batched_walk_pos, batched_edge_index, batch_index


