# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:37:52 2025

@author: sergi
"""

## have a need for a series of functions that will map actions to designs
## in the meep simulation environment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

## TODO: if using tanh, I have set to > 0, but if I switch ti sigmoid, it'd be 0.5 or 0.6

def decode_actions_to_design(actions, min_diam=0.04, max_diam=0.16, step=0.02):
    """
    Maps a 2-channel action vector per node to binary hole presence and hole diameter.

    Parameters:
        actions (Tensor): [num_nodes, 2] tensor from actor. Columns: [presence_score, size_score]
        min_diam (float): Minimum hole diameter (in microns)
        max_diam (float): Maximum hole diameter (in microns)
        step (float): Fabrication grid step for rounding (e.g., 0.02 um)

    Returns:
        Tuple of (hole_flag, diameters):
            hole_flag: (num_nodes,) binary tensor (0 or 1)
            diameters: (num_nodes,) float tensor (diameter in microns)
    """
    presence_score = actions[:, 0]
    size_score = actions[:, 1]

    # Binary presence decision (threshold at 0.6 for sigmoid, 0 for tanh)
    hole_flag = (presence_score > 0.6).float() # depends on whether tanh or sigmoid

    # Scale size_score to [min_diam, max_diam]: current one assumes tanh, if sigmoid, change formula
    # scaled_size = ((size_score + 1) / 2) * (max_diam - min_diam) + min_diam
    scaled_size = size_score * (max_diam - min_diam) + min_diam

    # best to have a more robust quantization procedure
    n_bins = int(round((max_diam - min_diam) / step)) + 1
    # convert to bin index, clamp, then map back
    idx = torch.round((scaled_size - min_diam) / step)
    idx = torch.clamp(idx, 0, n_bins - 1)
    quantized = min_diam + idx * step
    # Apply fabrication rounding
    # scaled_size = torch.round(scaled_size / step) * step

    # Zero out diameters where no hole is placed
    final_diameters = hole_flag * quantized 

    return hole_flag, final_diameters

def topology_matrix_from_decoded_actions(hole_flag, diameters, shape=(30,14)):
    '''
    Create a binary design matrix from the original design in dataset after [actions] are performed on it. 
    '''
    device = hole_flag.device
    matrix = torch.zeros(shape, dtype=torch.float32, device=device)
    idx = 0
    # seems largely unnecessary, just reshape hole_flag and turn to numpy? 
    for i in range(shape[0]):
        for j in range(shape[1]):
            if hole_flag[idx] > 0:
                matrix[i,j] = diameters[idx] # only if we are saving the diameters like that, otherwise binary
                # matrix[i,j] = 1 # if using above comment this one out 
            idx += 1
    return matrix.cpu().numpy()

# for halving without changing models
def mirror_design(hole_flag_half, diameters_half, symmetry='x', design_shape=(30,14)):
    H, W = design_shape
    
    if symmetry == 'x':
        W_half = W // 2
        hole_flag_half = hole_flag_half.view(H, W_half)
        diameters_half = diameters_half.view(H, W_half)

        hole_flag_full = torch.cat([hole_flag_half, torch.flip(hole_flag_half, dims=[1])], dim=1)
        diameters_full = torch.cat([diameters_half, torch.flip(diameters_half, dims=[1])], dim=1)

    elif symmetry == 'y':
        H_half = H // 2
        hole_flag_half = hole_flag_half.view(H_half, W)
        diameters_half = diameters_half.view(H_half, W)

        hole_flag_full = torch.cat([hole_flag_half, torch.flip(hole_flag_half, dims=[0])], dim=0)
        diameters_full = torch.cat([diameters_half, torch.flip(diameters_half, dims=[0])], dim=0)

    else:
        raise ValueError("Unsupported symmetry type. Use 'x' or 'y'")

    return hole_flag_full.flatten(), diameters_full.flatten()

