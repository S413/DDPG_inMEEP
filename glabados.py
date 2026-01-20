# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:18:50 2025

@author: sergi
"""

# building some functions and hashes to keep the design simulation results
# under the current simulation settings -- speed-up

import hashlib
import numpy as np
import json 
import pathlib
import torch

from simulation_funcs import run_meep_sim_wsl

def hash_design_quantized(hole_flag: torch.Tensor,
                      diam_um: torch.Tensor,
                      step_um: float = 0.02,
                      sim_config: dict | None = None) -> str:
    """
    Deterministic hash of a design from torch tensors.
    - hole_flag: 0/1 tensor, any dtype, any device, same shape as diam_um
    - diam_um: diameters in microns (float), same shape
    - step_um: quantization step (e.g., 0.02 µm -> codes 0,1,2,...)
    - sim_config: optional dict of sim params to fold into the key
    """
    if hole_flag.shape != diam_um.shape:
        raise ValueError(f"Shape mismatch: {hole_flag.shape=} vs {diam_um.shape=}")

    # Detach, move to CPU, make contiguous
    hf = hole_flag.detach().to(dtype=torch.uint8).contiguous().view(-1).cpu()
    d  = diam_um.detach().to(dtype=torch.float32).contiguous().view(-1).cpu()

    # Quantize to integer codes; mask non-holes to 0
    # e.g., 0.00→0, 0.04→2, 0.16→8 when step=0.02
    codes = torch.round(d / step_um).to(torch.int16)
    codes = torch.where(hf.bool(), codes, torch.zeros_like(codes))

    # Pack bytes (explicit little-endian for cross-platform stability)
    hf_np    = hf.numpy()                                         # uint8
    codes_np = codes.numpy().astype('<i2', copy=False)            # int16 little-endian
    shape_np = np.array(diam_um.shape, dtype='<i4')

    h = hashlib.sha256()
    h.update(b"SCHEMA:2;UNIT=um;STEP=" + repr(float(step_um)).encode())
    h.update(b";SHAPE:")
    h.update(shape_np.tobytes())
    if sim_config is not None:
        h.update(b";CFG:")
        h.update(json.dumps(sim_config, sort_keys=True).encode())

    h.update(b";HF:")
    h.update(hf_np.tobytes())
    h.update(b";DQ:")
    h.update(codes_np.tobytes())
    return h.hexdigest()

class DesignCache:
    def __init__(self, cache_file):
        self.cache_file = pathlib.Path(cache_file)
        self.cache = {}
        if self.cache_file.exists():
            self.load()
            
    def load(self):
        with open(self.cache_file, 'r') as f:
            self.cache = json.load(f)
            
    def save(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
            
    def get(self, design_hash):
        return self.cache.get(design_hash, None)
    
    def add(self, design_hash, transmissions):
        self.cache[design_hash] = transmissions
        
def simulation_cacher(cache_obj, hole_flag, diameters):
    '''
    We will first check the cache for the results, else we run a simulation in MEEP
    Recall that is is only valid for the current simulation setup. As soon as you change
    a parameter in the simulation script, this is not guaranteed valid anymore
    '''  
    design_hash = hash_design_quantized(hole_flag, diameters)
    # sanity check
    print("Cache key:", (design_hash))

    cached_result = cache_obj.get(design_hash)
    
    if cached_result:
        print("Cache hit.")
        return cached_result
    else:
        print("Cache miss. Simulating...")
        Tt, Tb = run_meep_sim_wsl(hole_flag, diameters)
        cache_obj.add(design_hash, (Tt, Tb))
        cache_obj.save()
        return Tt, Tb