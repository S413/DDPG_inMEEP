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
import numpy as np

## TODO: if using tanh, I have set to > 0, but if I switch ti sigmoid, it'd be 0.5 or 0.6

def decode_actions_to_design(
    actions,
    min_diam=0.04,
    max_diam=0.16,
    step=0.02,
    *,
    presence_activation="sigmoid",   # "sigmoid" | "tanh" | "raw"
    presence_thresh=0.6,             # if "tanh" use ~0.0; if "raw", pick accordingly
    size_activation="raw",          # "sigmoid" | "tanh" | "raw"
    quant_mode="stochastic",            # "nearest" | "floor" | "stochastic"
    eps=1e-9
):
    """
    Deterministic (or stochastic) binning from continuous actions to a fabrication grid.
    Returns:
        hole_flag: (N,) float tensor in {0., 1.}
        final_diameters: (N,) float tensor in microns, 0.0 where no hole
    """

    presence_score = actions[:, 0]
    size_score = actions[:, 1]

    # Sanity checks 
    print(f"Largest raw size: {size_score.max()}")
    print(f"Smallest raw size: {size_score.min()}")
    print(f"Entire sizes array: {size_score}")

    # ---- 1) Presence
    if presence_activation == "sigmoid":
        p = torch.sigmoid(presence_score)
    elif presence_activation == "tanh":
        p = (torch.tanh(presence_score) + 1) * 0.5
    elif presence_activation == "raw":
        # Map raw roughly into [0,1] with a squash; change if you truly want raw
        p = torch.sigmoid(presence_score)
    else:
        raise ValueError("presence_activation must be 'sigmoid'|'tanh'|'raw'.")

    hole_flag = (p > presence_thresh).to(actions.dtype)

    # ---- 2) Size normalization s in [0,1]
    if size_activation == "sigmoid":
        s = torch.sigmoid(size_score)
    elif size_activation == "tanh":
        s = (torch.tanh(size_score) + 1) * 0.5
    elif size_activation == "raw":
        # Hard clamp raw to [0,1] to avoid runaway
        s = torch.clamp(size_score, 0.0, 1.0)
    else:
        raise ValueError("size_activation must be 'sigmoid'|'tanh'|'raw'.")

    # ---- 3) Build the *actual* grid we will snap to
    # Use floor so we don't exceed max even if (range/step) is non-integer
    total_range = max_diam - min_diam
    n_bins = int(torch.floor(torch.tensor(total_range / step)) .item()) + 1
    grid = min_diam + torch.arange(n_bins, device=actions.device, dtype=actions.dtype) * step
    # last grid point <= max_diam by construction; we can optionally append max_diam if close:
    if grid[-1] < max_diam - eps:
        # append max_diam as a final catch-all exact bin
        grid = torch.cat([grid, torch.tensor([max_diam], device=grid.device, dtype=grid.dtype)])
        n_bins = grid.numel()

    # ---- 4) Map s in [0,1] to a *real-valued* diameter, then quantize to grid
    # Use arithmetic mapping only to pick an index; final value is from `grid` to avoid drift.
    # Compute ideal (unquantized) target diameter:
    target = min_diam + s * (max_diam - min_diam)

    # Convert target to a bin index
    raw_idx = (target - min_diam) / (grid[1] - grid[0]).clamp_min(eps)  # nominal step (first gap)
    raw_idx = torch.clamp(raw_idx, 0, n_bins - 1 - eps)  # stay within valid half-open range

    if quant_mode == "nearest":
        idx = torch.round(raw_idx)
    elif quant_mode == "floor":
        idx = torch.floor(raw_idx)
    elif quant_mode == "stochastic":
        # Stochastic rounding between floor and ceil
        lo = torch.floor(raw_idx)
        hi = torch.clamp(lo + 1, max=n_bins - 1)
        prob_hi = (raw_idx - lo).clamp(0, 1)
        idx = torch.where(torch.rand_like(prob_hi) < prob_hi, hi, lo)
    else:
        raise ValueError("quant_mode must be 'nearest'|'floor'|'stochastic'.")

    idx = idx.to(torch.long)
    quantized = grid[idx]  # exact grid value; idempotent on re-decode

    # ---- 5) Mask by presence (diameter=0 where no hole)
    final_diameters = hole_flag * quantized

    # sanity checks 2
    print(f"Largest final diameter: {final_diameters.max()}")
    print(f"Smallest final diameter: {final_diameters.min()}")
    print(f"Entire final diameters array: {final_diameters}")

    return hole_flag, final_diameters

def topology_matrix_from_decoded_actions(hole_flag, diameters, shape=(30,14)):
    '''
    Create a binary design matrix from the original design in dataset after [actions] are performed on it. 
    '''
    device = hole_flag.device
    matrix = torch.zeros(shape, dtype=torch.float32, device=device)
    idx = 0
    # seems largely unnecessary, just reshape hole_flag and turn to numpy? 
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         if hole_flag[idx] > 0:
    #             matrix[i,j] = diameters[idx] # only if we are saving the diameters like that, otherwise binary
    #             # matrix[i,j] = 1 # if using above comment this one out 
    #         idx += 1
    matrix = hole_flag.view(shape[0], shape[1]) * diameters.view(shape[0], shape[1])
    # sanity check, resulting matrix:
    print(f"Resulting design matrix cacher (diameters, 0 where no hole):\n{matrix}")
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

