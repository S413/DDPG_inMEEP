# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:09:52 2025

@author: sergi
"""

## functions for the simulation part of the DDPG actor critic using GCN

## there will be a difference if code run from linux computer and not original windows pc 

import json
import subprocess
import pathlib
import uuid
import numpy as np
import platform
import os 
import shutil

# some helper functions to detect OS and convert paths 
def _is_windows():
    return platform.system() == "Windows"

def _is_linux():
    return platform.system() == "Linux"

def _is_wsl_linux():
    try:
        return _is_linux() and "microsoft" in platform.release().lower()
    except Exception:
        return False
    
def _has_wsl_on_windows():
    return _is_windows() and shutil.which("wsl") is not None

def _win_to_wsl_path(win_path: str):
    # convert win path to linux path 
    s = win_path.replace("\\", "/")
    
    # Handle UNC paths like //wsl$/Ubuntu/... or \\wsl$\Ubuntu\...
    if s.startswith("//wsl$/") or s.startswith("/wsl$/"):
        # Extract the distro and path: //wsl$/Ubuntu/path/to/file -> /path/to/file
        parts = s.split("/")
        # parts[0] = '', parts[1] = '', parts[2] = 'wsl$', parts[3] = 'Ubuntu', parts[4:] = rest
        if len(parts) > 4:
            return "/" + "/".join(parts[4:])
    
    # Handle regular Windows paths like C:\path\to\file
    if len(s) > 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:] if s[2] == "/" else s[2:]
        return f"mnt/{drive}{rest}"
    
    return s

def run_meep_sim_wsl(hole_flag, 
                     diameters,
                     mode: str = "auto",
                     wsl_distro: str = "Ubuntu",
                     ):
    uid = uuid.uuid4().hex
    
    # first decided on runtime mode
    if mode == "auto":
        if _has_wsl_on_windows():
            mode_resolved = "windows_wsl"
            # define tmp directory here since it is windows and might not open if in wsl path
            TMP_DIR = pathlib.Path(r"\\wsl$\Ubuntu\root\QRLike\DDPG_Agent\tmp")
        elif _is_linux():
            mode_resolved = "linux"
            # define tmp directory here since it is linux and no need to convert or go to externals
            TMP_DIR = pathlib.Path("/home/sergio/MeepProj/DDPG_inMEEP/tmp")
        else:
            raise RuntimeError("Could not auto-detect runtime mode. Please specify 'mode' explicitly.")
    else:
        mode_resolved = mode

    # now we deal with the temp directory and temp files 
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    param_file = TMP_DIR / f"params_{uid}.json"
    result_file = TMP_DIR / f"result_{uid}.json"

    param_dict = { 
        "hole_flag": hole_flag.detach().cpu().numpy().tolist(),
        "diameters": diameters.detach().cpu().numpy().tolist()
        }
    
    with open(param_file, "w") as f:
        json.dump(param_dict, f)
    
    try:
        if mode_resolved == "windows_wsl":
            # convert windows paths to wsl readable paths
            param_file_wsl = _win_to_wsl_path(str(param_file))
            result_file_wsl = _win_to_wsl_path(str(result_file))

            # defining python and script path here since it is windows -- these are the paths within wsl
            python_path = "/root/miniconda3/envs/mp/bin/python"
            script_path = "/root/QRLike/simulate_in_wsl.py"

            cmd = [
                "wsl", "-d", wsl_distro,
                python_path, script_path,
                "--param_file", param_file_wsl,
                "--out_file", result_file_wsl,
                "--template", "2x2"
            ]
            subprocess.run(cmd, check=True) 

            # read back into windows
            with open(result_file, "r") as f:
                result = json.load(f)

        elif mode_resolved == "linux":
            # use linux paths directly
            # some of the paths might also differ

            python_path = "/home/sergio/anaconda3/envs/mp/bin/python" # not sure what the python path will be in the server. Do that first.
            script_path = "/home/sergio/MeepProj/DDPG_inMEEP/simulate_in_wsl.py"

            cmd = [
                python_path, script_path,
                "--param_file", str(param_file),
                "--out_file", str(result_file)
            ]
            subprocess.run(cmd, check=True)

            with open(result_file, "r") as f:
                result = json.load(f)

        else:
            raise ValueError(f"Unknown mode '{mode_resolved}'")
        
    finally:
        # clean up temp files
        if param_file.exists():
            param_file.unlink()
        if result_file.exists():
            result_file.unlink()

    return result['Transmission']