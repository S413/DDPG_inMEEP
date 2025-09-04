# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:40:27 2025

@author: sergi
"""

## the GCN Actor Critic Networks
## I should keep in mind this is an optim to some design: I suggest even
## reworking 2025.08.13 because adding edge awareness and what not...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_add_pool # global_mean_pool ? diff ?

class GCNActor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCNActor, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # output decoder: different per node ( in this case assume uniform )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            # nn.Tanh() # outputs in range [-1,1] like in the paper. Must still map and scale after
            nn.Sigmoid() # outputs in range [0,1] unlike paper, but closer to meep binary reqs
            ) # the output vector should be (num_nodes, out_channels), each row is a node in graph
        
    def forward(self, x, edge_index):
        for conv in self.gcn_layers:
            x = F.relu(conv(x, edge_index))
        return self.decoder(x)
    
# class GCNCritic(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers=3):
#         super(GCNCritic, self).__init__()
#         self.gcn_layers = nn.ModuleList()
#         self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
#         self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        
#         self.global_fc = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, 1) # predicted reward: scalar
#             )
        
#     def forward(self, x, edge_index):
#         for conv in self.gcn_layers:
#             x = F.relu(conv(x, edge_index))
#         x = torch.mean(x, dim=0)
#         return self.global_fc(x)

# class GCNCritic(nn.Module):
#     def __init__(self, in_channels, hidden_channels, action_dim):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # after global pooling we’ll concat with the action
#         self.lin1 = nn.Linear(hidden_channels + action_dim, hidden_channels)
#         self.lin2 = nn.Linear(hidden_channels, 1)

#     def forward(self, x, edge_index, action):
#         # 1) extract a graph-level embedding
#         h = F.relu(self.conv1(x, edge_index))
#         h = F.relu(self.conv2(h, edge_index))
#         h = torch.mean(h, dim=0, keepdim=True)      # simple mean-pool

#         print("h.shape:", h.shape)
#         print("action.shape:", action.shape)


#         # 2) append the action
#         ha = torch.cat([h, action], dim=-1)

#         # 3) value head
#         q = F.relu(self.lin1(ha))
#         q = self.lin2(q)
#         return q

class GCNCritic(nn.Module):
    def __init__(self, in_channels, hidden_channels, action_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels) # 3,64
        self.conv2 = GCNConv(hidden_channels, hidden_channels) #64, 64

        # node-level MLP
        self.lin1 = nn.Linear(hidden_channels + action_dim, hidden_channels) # (274,64)
        self.lin2 = nn.Linear(hidden_channels, 1) # (64, 1)

    def forward(self, x, edge_index, action):
        """
        x      : [210,3] # for coordinates x,y and hole presence bit
        action : [210,2] # for hole presence bit and hole diameter  
        """
        h = F.relu(self.conv1(x, edge_index)) # 210,3 2,1462 -> 210,64
        h = F.relu(self.conv2(h, edge_index))          # h is above, edge is 2,1462 -> 210,64

        if h.shape[0] != action.shape[0]:
            raise ValueError(f"action expects {h.shape[0]} rows, got {action.shape[0]}")

        ha = torch.cat([h, action], dim=-1)            # [N, H+A]
        q_node = F.relu(self.lin1(ha))                 # [N, H]
        q_node = self.lin2(q_node)                    # [N, 1]

        q_graph = q_node.mean(dim=0, keepdim=True)     # [1, 1]
        return q_graph

    
# we are assuming input node features = 3

###############################################################################
###############################################################################

class EdgeAwareCritic(nn.Module):
    def __init__(self, in_channels, action_dim, hidden_dim, edge_dim=None):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)         # <-- NEW
        self.s1 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.s2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.a_proj = nn.Linear(action_dim, hidden_dim)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta  = nn.Linear(hidden_dim, hidden_dim)
        self.sa    = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.q1 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, action):
        h = F.relu(self.in_proj(x))                               # 9 → 128
        h = F.relu(self.s1(h, edge_index, edge_attr))
        h = F.relu(self.s2(h, edge_index, edge_attr))
        a = F.relu(self.a_proj(action))
        h = self.gamma(a) * h + self.beta(a)
        h = F.relu(self.sa(h, edge_index, edge_attr))
        h = F.relu(self.q1(h))
        q_node = self.q2(h)
        return q_node.mean(dim=0, keepdim=True)
    
class EdgeAwareActor(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, edge_dim=None, num_layers=3):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)         # <-- NEW

        def make_conv(ic, oc):
            return (GATv2Conv(ic, oc, heads=2, concat=False, edge_dim=edge_dim)
                    if edge_dim is not None else
                    GATv2Conv(ic, oc, heads=2, concat=False))

        self.convs = nn.ModuleList()
        self.convs.append(make_conv(hidden_dim, hidden_dim))      # <-- take hidden_dim as input
        for _ in range(num_layers - 1):
            self.convs.append(make_conv(hidden_dim, hidden_dim))

        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.in_proj(x))                               # <-- project 9 → 128
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))            # <-- conv expects 128 in
        return torch.sigmoid(self.dec(h))