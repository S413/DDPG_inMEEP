# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:09:52 2025

@author: sergi
"""

## functions for the simulation part of the DDPG actor critic using GCN

import json
import subprocess
import pathlib
import uuid
import numpy as np

def run_meep_sim_wsl(hole_flag, diameters):
    uid = uuid.uuid4().hex
    
    TMP_DIR = pathlib.Path(r"C:\tmp")          # pick any writable directory
    TMP_DIR.mkdir(parents=True, exist_ok=True) # ensure it exists
    
    param_file_win = TMP_DIR / f"params_{uid}.json"
    result_file_win = TMP_DIR / f"result_{uid}.json"

    param_dict = {
        "hole_flag": hole_flag.tolist(),
        "diameters": diameters.tolist()
    }

    # Save parameters to file
    with open(param_file_win, "w") as f:
        json.dump(param_dict, f)

    # Convert to WSL paths
    param_file_wsl = f"/mnt/c/tmp/{param_file_win.name}"
    result_file_wsl = f"/mnt/c/tmp/{result_file_win.name}"

    # python path of the environment in use
    python_path = "/root/miniconda3/envs/mp/bin/python"
    
    # Run the WSL-side script
    cmd = [
        "wsl", "-d", "Ubuntu",  # 
        python_path, "/root/QRLike/simulate_in_wsl.py",
        "--param_file", param_file_wsl,
        "--out_file", result_file_wsl
    ]
    subprocess.run(cmd, check=True)

    # Read result back
    with open(result_file_win, "r") as f:
        result = json.load(f) # this is a dict {"transmission":([top trans], [bot trans])}
    
    # clear to prevent clutter
    param_file_win.unlink()
    result_file_win.unlink()
    
    return result["Transmission"]