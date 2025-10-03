# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:30:07 2025

@author: sergi
"""

## I'll be trying the RL approach with the graph neural network
## Therefore, I'll need to transform my inputs into a graph setup for input
## @today 2025/08/12 adding some other features to the graph built. see if it helps

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected 
import numpy as np
import math

def create_graph_from_topology_matrix(matrix, 
                                      use_8_conn=False, 
                                      include_coords=True,
                                      sinusodal_posenc=True,
                                      add_boundary_flags=True,
                                      add_selfloops=True,
                                      undirected=True):
    '''
    Converts binary topolgy matric into a Pytorch Geometric Graph
    This is still not a great way to do this, I'm not convinced'

    Parameters
    ----------
    matrix : TYPE
        DESCRIPTION.
    use_8_conn : TYPE, optional
        DESCRIPTION. The default is False.
    include_coords : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    
    H, W = matrix.shape # unsure if they will be padded or unpadded. Also helps if I change design region dimensions, I spose
    node_features = []
    edge_src = []
    edge_dst = []
    edge_attrs = []
    
    def node_id(i, j):
        return i * W + j
    
    # helper fun for normalized positions in [-1,1]
    def norm_xy(i, j):
        x = 0.0 if W == 1 else (j / (W-1)) * 2.0 - 1.0
        y = 0.0 if H == 1 else (i / (W-1)) * 2.0 - 1.0
        return x,y
    
    # neighbors patterns 
    neighbors = [(-1,0), (0,-1), (1,0), (0,1)] # basically up,left,down,right
    if use_8_conn:
        neighbors += [(-1,-1), (-1,1), (1,-1), (1,1)] # adding the diagonal directions
    
    # precompute some masks here
    def is_edge_cell(i, j):
        return i == 0 or i == H-1 or j == 0 or j == W-1
    def is_corner_cell(i, j):
        return (i in (0, H-1)) and (j in (0, W-1))
    def is_port_cell(i, j):
        # replace with real port cells, keep-out zones. Might not use this
        return 0 # place holder for now
    
    # build node features and edges. Some of these have changed, and the feature order has changed also
    for i in range(H):
        for j in range(W):
            hole_flag = float(matrix[i, j])
            feats = [hole_flag]
            
            x_norm, y_norm = norm_xy(i, j)
            if include_coords:
                feats += [x_norm, y_norm] # appendiong to feats list
                if sinusodal_posenc:
                    feats += [math.sin(math.pi * x_norm),
                              math.cos(math.pi * x_norm),
                              math.sin(math.pi * y_norm),
                              math.cos(math.pi * y_norm)]
            if add_boundary_flags:
                feats += [float(is_edge_cell(i, j)),
                          float(is_corner_cell(i,j)),
                          # float(is_port_cell(i,j)) # might not want to add this atm
                          ]
            node_features.append(feats)
            
            # edges from (i,j) to neighbors
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    u = node_id(i, j)
                    v = node_id(ni, nj)
                    edge_src.append(u)
                    edge_dst.append(v)
                    
                    x2, y2 = norm_xy(ni,nj)
                    dx, dy = (x2 - x_norm), (y2 - y_norm)
                    r = math.hypot(dx, dy)
                    is_diag = 1.0 if (di != 0 and dj != 0) else 0.0
                    edge_attrs.append([dx, dy, r, is_diag])
                    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # make undirected, keep node attrs in sync, whatever that latter half means
    if undirected:
        # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        # duplicate attrs for reverse edges
        # to undirected doubles edges; mirror attrs by simple title
        # edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = to_undirected(
                edge_index, edge_attr=edge_attr, num_nodes=x.size(0)
                )   
        
    if add_selfloops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # for self loops define dx=dy=0, r=0, is_diag=0
        num_loops_added = x.size(0)
        self_attr = torch.zeros((num_loops_added, edge_attr.size(1)), dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, self_attr], dim=0)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # can also stash masks for actor/critic
    # look into that later
    
    return data


def main():
    # then I guess we give it a try
    
    # load the matrices
    labels = np.load('labels.npy')
    designs = np.load('FullDesign.npy').reshape((-1, 32,32,1))
    
    # now create the graph from a desing and check contents
    graph_test = create_graph_from_topology_matrix(designs[100].squeeze(), use_8_conn=False, include_coords=True) # the default settings False, 


if __name__ == '__main__':
    main()
