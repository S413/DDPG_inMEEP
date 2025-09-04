# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:18:45 2025

@author: sergi
"""

# just testing the agent model after minimal training
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch 
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt
import pickle 

# imports from other files for RL and graph
from RL_help_funcs import GaussianNoise, reward_balanced_transmission, ReplayBuffer, reward_min, OUNoise, noise_scale_scheduler
from action_to_designs import decode_actions_to_design, topology_matrix_from_decoded_actions, mirror_design
from GCN_ActorCritic import GCNActor, GCNCritic, EdgeAwareActor, EdgeAwareCritic
from turnToGraph import create_graph_from_topology_matrix
from glabados import DesignCache, simulation_cacher

designs = np.load('FullDesign.npy').reshape(-1, 32, 32)
# if you're going to unpad them, do so here
designs = designs[:,1:31,9:23] # now shape should be (N, 30, 14)
designs = designs[:, :, :7] # half, then we mirror to full
graph_list = [create_graph_from_topology_matrix(d, use_8_conn=True, add_selfloops=False) for d in designs]
print("Data loaded and turned to graph representration.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

action_dim = (210,2) # number of nodes, features per node [hole, diameter]
 
# randomly grab initial design. 
initial_graph = random.choice(graph_list).to(device) # does this matter more than I think? 

# must still refine hidden size and output size for these models
actor = EdgeAwareActor(in_channels=initial_graph.x.size(1), 
                        hidden_dim=128, 
                        out_channels=2, 
                        edge_dim=initial_graph.edge_attr.size(1), num_layers=5).to(device)
critic = EdgeAwareCritic(in_channels=initial_graph.x.size(1), 
                          action_dim=action_dim[1], hidden_dim=128, 
                          edge_dim=initial_graph.edge_attr.size(1)).to(device)


loaded_actor = torch.load('models/Second_Great_Convergence/actor_latest.pt')
loaded_critic = torch.load('models/Second_Great_Convergence/critic_latest.pt')

actor.load_state_dict(loaded_actor)
critic.load_state_dict(loaded_critic)

# load cache
cache = DesignCache('./CacheLocation/cache.json')

# start the looping from here then, and maybe keep a list of the created designs for plotting
gen_des_list = []

# retrieve the inital design image
initial_design_half = initial_graph.x[:,0].cpu()
initial_design_full = mirror_design(initial_design_half, initial_design_half)[0]
gen_des_list.append(initial_design_full.cpu().reshape(30,14))

# probably also need some trigger to decide when to stop
o_total_T = 0 # if this falls from one generation to the next, we stop. hopefully it goes up only
o_delta_T = 1.0 # if this grows from one generation to the next, we stop. hopefully it goes down only

# save transmissions at 1550 nm for plotting the evolution of the transmission profile of the designs
# just realized we do not have the transmission values for the initial design ...
transT = []
transB = []

# obtain the transmission profile of the initial design? 
Tt, Tb = simulation_cacher(cache, initial_design_full.reshape(30*14), initial_design_full.reshape(30*14))
transT.append(Tt[-2])
transB.append(Tb[-2])

for _ in range(7):
    # generate actions
    actions = actor(initial_graph.x, initial_graph.edge_index, initial_graph.edge_attr)
    hole_flag, diameters = decode_actions_to_design(actions)
    hole_flag_full, diameters_full = mirror_design(hole_flag, diameters)
    
    gen_des_list.append(hole_flag_full.detach().cpu().numpy().reshape(30,14))
    
    # should also check the transmission profile of the output design
    # done through glabados so we have access to the stored designs transmission profiles
    Tt, Tb = simulation_cacher(cache, hole_flag_full, diameters)
    reward = (
        # -0.2 * (step+1) + # we want to get design fast
        # 2.0 * (Tt[-2] + Tb[-2]) + # high transmission is good
        -2.0 * abs(Tt[-2]-Tb[-2]) # large difference in transission per arm is bad
        )
    transT.append(Tt[-2])
    transB.append(Tb[-2])
    updates_design = topology_matrix_from_decoded_actions(hole_flag_full, diameters_full)
    # we have to create the graph representation of the next design, doughnut, same params
    next_graph = create_graph_from_topology_matrix(updates_design[:,:7], True, True).to(device)
    
    initial_graph = next_graph
    
    if o_delta_T < abs(Tt[-2]-Tb[-2]):
        o_delta_T = abs(Tt[-2]-Tb[-2])

# now we create the plots
# first plot the transmission profile evolution of the designs @ ~1550 nm
plt.figure()
plt.title("Output Arm Normalized Transmission @ ~1550nm per Design")
plt.xlabel("Design No.")
plt.ylabel("Normalized Transmission")
plt.plot(transT, label="Top Arm")
plt.plot(transB, label="Bot Arm")
plt.plot(np.add(transB, transT), label="Total")
plt.legend()
plt.show()

# now we plot the design evolution thorugh the images
fig, axes = plt.subplots(1, len(gen_des_list)*2-1, figsize=(4*len(gen_des_list), 4))
for i, img in enumerate(gen_des_list):
    # Plot image
    ax_img = axes[i*2]
    ax_img.imshow(img)
    ax_img.axis("off")
    
    # Add arrow in between images
    if i < len(img)-1:
        ax_arrow = axes[i*2 + 1]
        ax_arrow.axis("off")
        ax_arrow.annotate("",
                          xy=(1, 0.5), xycoords="axes fraction",
                          xytext=(0, 0.5), textcoords="axes fraction",
                          arrowprops=dict(arrowstyle="->", lw=2))

plt.tight_layout()
plt.show()