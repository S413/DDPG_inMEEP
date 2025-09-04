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

from simulation_funcs import run_meep_sim_wsl
from action_to_designs import topology_matrix_from_decoded_actions

def hash_design(matrix):
    ''' 
    ''' 
    matrix_bytes = matrix.astype(np.uint8).tobytes()
    return hashlib.sha256(matrix_bytes).hexdigest()

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
    matrix = topology_matrix_from_decoded_actions(hole_flag, diameters)
    design_hash = hash_design(matrix)
    
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