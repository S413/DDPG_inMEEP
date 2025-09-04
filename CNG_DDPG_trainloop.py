# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:19:51 2025

@author: sergi
"""

## here is the full train loop for DDPG using the graph network.

# TODO: adding variable hole size (already mostly in, just have to take into acount and scale)
# TODO: adding other design parameters (length and width of design region) -- requires more work -- limits?

# imports
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
from RL_help_funcs import GaussianNoise, reward_balanced_transmission, PriorityReplayBuffer, reward_min, OUNoise, noise_scale_scheduler
from action_to_designs import decode_actions_to_design, topology_matrix_from_decoded_actions, mirror_design
from GCN_ActorCritic import GCNActor, GCNCritic, EdgeAwareActor, EdgeAwareCritic
from turnToGraph import create_graph_from_topology_matrix
from glabados import DesignCache, simulation_cacher

# import from other files for the MEEP simulation
from simulation_funcs import run_meep_sim_wsl 

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# train function loop for ddpg
def train_ddpg(graph_list,
               epochs=1000,
               steps_per_epoch=1, # this should be the max episode length in any case
               batch_size=64,
               gamma=0.99,
               tau=0.005,
               lr_actor=1e-4,
               lr_critic=1e-3,
               update_every=8, # do gradient step N what? episodes or actions taken (across episodes)?
               buffer_capacity=10000,
               eval_interval=1
               ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    initial_graph = random.choice(graph_list) # randomly select an initial design from list 
    
    action_dim = (210,2) # number of nodes, features per node [hole, diameter]
    
    actor = EdgeAwareActor(in_channels=initial_graph.x.size(1), 
                           hidden_dim=128, 
                           out_channels=2, 
                           edge_dim=initial_graph.edge_attr.size(1), num_layers=5).to(device)
    critic = EdgeAwareCritic(in_channels=initial_graph.x.size(1), 
                             action_dim=action_dim[1], hidden_dim=128, 
                             edge_dim=initial_graph.edge_attr.size(1)).to(device)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    optim_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    
    # exploration noise
    noise = OUNoise(shape=(210,2), mu=0.0, theta=0.15, sigma=0.2, scale=2.0)# good results so far
    buffer = PriorityReplayBuffer(buffer_capacity, initial_mode='td_error')
    
    # tensorboard's writer defined here
    writer = SummaryWriter(log_dir='runs/ddpg_training/new_model_new_graph03')
    
    # global steps or total number of steps
    total_steps = 0 # count environment interactions
    
    # ewma_reward, for keeping track of a trend and therefore whether or not this shite is learning
    ewma_reward = 0
    
    # init cache object and its directory
    cache = DesignCache('./CacheLocation/cache.json')
    
    # best totals (for keeping track of best T and smallest deltaT)
    best_T = 0.6 # largest default total transmission
    delta_T = 0.06 # smallest difference between Tt and Tb transmission

    for epoch in range(epochs):
        # some things need resetting per episode
        episode_reward = 0
        episode_length = 0 # should end episode once we have a nice enough transmission profile, no? how many steps did it take to arrive there?
        next_graph = random.choice(graph_list) # don't want to always start from same design. P_0
        good_enough_flag = False
        bonus = 0
        noise.reset()
        noise.scale = noise_scale_scheduler(epoch,
                                            initial_scale=2.0, decay_rate=0.5,
                                            min_scale=0.05) # best test some jazz
        # just to know what the highest tracked values as so far
        print(f"Best total transmission for epoch {epoch}: {best_T}")
        print(f"Best minimal difference for epoch {epoch}: {delta_T}")
        for step in range(steps_per_epoch):
            total_steps += 1 # start counting global steps
            # ── Select action ──────────────────────────
            print(f"Epoch:{epoch}, step:{step}")
            # graph = random.choice(graph_list).to(device)
            graph = next_graph.to(device) # this should be initial graph on first step and then whatever next graph is after action
            actor.eval()
            with torch.no_grad():
                action = actor(graph.x, graph.edge_index, graph.edge_attr)
            action = action + noise.sample().to(device)
            action = torch.clamp(action, -1, 1)

            # ── Decode & simulate ──────────────────────
            hole_flag, diameters = decode_actions_to_design(action) # equivalent to next state, with T
            hole_flag_full, diameters_full = mirror_design(hole_flag, diameters) # mirroring if halved
            # keep the fully worked action that gets used in MEEP, but halved because half
            executed_action = torch.stack([
                torch.tensor(hole_flag_full.reshape(30,14)[:,:7].reshape(-1), dtype=torch.float, device=device),
                torch.tensor(diameters_full.reshape(30,14)[:,:7].reshape(-1), dtype=torch.float, device=device)
                ], dim=1) # (2,30,7) -> flatten with reshape(-1), so we have (N,2)
            Tt, Tb = simulation_cacher(cache, hole_flag_full, diameters) # they are returned as lists at this point
            # make them into tensors or next function fails
            Tt = torch.Tensor(Tt)
            Tb = torch.Tensor(Tb)

            # take only the frequency closest to 1550 nm ? [-2] for width of 0.04 @ 1550 center... but not too centered
            reward = (
                # -0.2 * (step+1) + # we want to get design fast
                # 2.0 * (Tt[-2] + Tb[-2]) + # high transmission is good
                -best_T + Tt[-2]+Tb[-2] + # if total transmission is high, difference will be 0 or +
                -2.0 * abs(Tt[-2]-Tb[-2]) # large difference in transission per arm is bad
                )

            # if next_graph = graph, it'd be stateless, which it's not, so create graph from new design after action
            updates_design = topology_matrix_from_decoded_actions(hole_flag_full, diameters_full)
            # we have to create the graph representation of the next design, doughnut, same params
            next_graph = create_graph_from_topology_matrix(updates_design[:,:7], True, True).to(device)

            # ───── check if episode termination criteria is met ─────
            if (torch.abs(Tt[-2] - Tb[-2]) <= delta_T) and (Tt[-2]+Tb[-2] >= best_T):
                if torch.abs(Tt[-2]-Tb[-2]) < delta_T :
                    delta_T = torch.abs(Tt[-2] - Tb[-2]) # updating min difference
                if Tt[-2]+Tb[-2] > best_T:
                    best_T = Tt[-2]+Tb[-2] # updating max transmission
                # episode is done, maybe refine? 
                good_enough_flag = True
                bonus = 10.0

            # ───── update episode length and running rewards ────
            episode_length += 1
            episode_reward += reward + bonus

            # ── Store in buffer ─────────────────────────
            buffer.push(graph, executed_action, reward.item(), next_graph, good_enough_flag)
            
            # ── Update networks ────────────────────────
            if len(buffer) >= batch_size:
                print(f"Buffer length: {len(buffer)}")
                print("Updating networks...")
                samples, indices, is_weights = buffer.sample(batch_size, beta=0.4)
                states, actions, rewards, next_states, dones = zip(*samples)
                # dones because I was doing some jazz wrong
                batch_dones = torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1) # B,1
                is_weights = is_weights.to(device).unsqueeze(1) # for shape [B,1] ? 

                batch_rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)  # [B, 1]

                q_target_list = []
                q_pred_list = []
                pred_actions_list = []

                for s, a, r, s_next, d in zip(states, actions, rewards, next_states, dones):
                    # Move graph data to device
                    s = s.to(device)
                    s_next = s_next.to(device)
                    a = a.to(device)
                    # done_mask = torch.tensor(d, dtype=torch.float, device=device).unsqueeze(1)

                    # --- Critic target ---
                    with torch.no_grad():
                        next_a = target_actor(s_next.x, s_next.edge_index, s_next.edge_attr)
                        q_tgt = target_critic(s_next.x, s_next.edge_index, s_next.edge_attr, next_a)  # [1, 1]
                        q_target_list.append(q_tgt)

                    # --- Critic prediction ---
                    q_pred = critic(s.x, s.edge_index, s.edge_attr, a)  # [1, 1]
                    q_pred_list.append(q_pred)

                    # --- Actor prediction (for actor loss) ---
                    pred_a = actor(s.x, s.edge_index, s.edge_attr)
                    pred_actions_list.append(pred_a)

                # Stack values
                q_target = torch.cat(q_target_list, dim=0)  # [B, 1]
                q_pred   = torch.cat(q_pred_list, dim=0)    # [B, 1]
                
                # err on the side o' caution
                actor.train()
                critic.train()
                
                # Compute losses
                y = batch_rewards + gamma * q_target * (1.0 - batch_dones)
                # critic_loss = F.mse_loss(q_pred, y) 
                # td error after critic loss ... ? 
                td_errors = (q_pred - y) # do not detach here
                critic_loss = (is_weights * td_errors ** 2).mean()
                
                # updating the priorities using td errors
                td_errors_np = td_errors.detach().cpu().squeeze().numpy()
                buffer.update_priorities(indices, td_errors_np)
                
                optim_critic.zero_grad()
                critic_loss.backward()
                optim_critic.step()

                # --- Actor update ---
                actor_loss_list = []
                for s, pred_a in zip(states, pred_actions_list):
                    s = s.to(device)
                    q_val = critic(s.x, s.edge_index, s.edge_attr, pred_a)  # [1, 1]
                    actor_loss_list.append(-q_val)
                actor_loss = torch.mean(torch.cat(actor_loss_list, dim=0))

                optim_actor.zero_grad()
                actor_loss.backward()
                optim_actor.step()

                # --- Target network soft update ---
                for tgt, src in zip(target_actor.parameters(), actor.parameters()):
                    tgt.data.mul_(1 - tau)
                    tgt.data.add_(tau * src.data)
                # Hard update
                for tgt, src in zip(target_critic.parameters(), critic.parameters()):
                    tgt.data.mul_(1 - tau)
                    tgt.data.add_(tau * src.data)
                    
                writer.add_scalar("Loss/critic", critic_loss.item(), total_steps)
                writer.add_scalar("Loss/actor", actor_loss.item(), total_steps)
                writer.add_scalar("Transmission/Tt", Tt[-2], total_steps)
                writer.add_scalar("Transmission/Tb", Tb[-2], total_steps)
                writer.add_scalar("Reward", reward, total_steps)
                
            if good_enough_flag:
                print(f"Episode Length: {episode_length}. Good enough: ({Tt[-2]},{Tb[-2]}), R:{reward}")
                break
        
        # pure evaluation
        next_graph = initial_graph.to(device) # start each evaluation form the same initial design
        if epoch % eval_interval == 0:
            tot_eval_reward = 0 
            eval_graph = next_graph
            good_enough_flag = False
            for step in range(steps_per_epoch):
                actor.eval()
                with torch.no_grad():
                    action = actor(eval_graph.x, eval_graph.edge_index, eval_graph.edge_attr)
                action = torch.clamp(action, -1, 1)
                
                hole_flag, diameters = decode_actions_to_design(action) # equivalent to next state, with T
                hole_flag_full, diameters_full = mirror_design(hole_flag, diameters) # mirroring if halved
                Tt, Tb = simulation_cacher(cache, hole_flag_full, diameters) # they are returned as lists at this point
                # make them into tensors or next function fails
                Tt = torch.Tensor(Tt)
                Tb = torch.Tensor(Tb)

                # take only the frequency closest to 1550 nm ? [-2] for width of 0.04 @ 1550 center... but not too centered
                reward = (
                    # -0.2 * (step+1) + # we want to get design fast
                    # 2.0 * (Tt[-2] + Tb[-2]) + # high transmission is good
                    -best_T + Tt[-2]+Tb[-2] +
                    -2.0 * abs(Tt[-2]-Tb[-2]) # large difference in transission per arm is bad
                    )

                # if next_graph = graph, it'd be stateless, which it's not, so create graph from new design after action
                updates_design = topology_matrix_from_decoded_actions(hole_flag_full, diameters_full)
                # we have to create the graph representation of the next design, doughnut, same params
                next_graph = create_graph_from_topology_matrix(updates_design[:,:7], True, True).to(device)
                if (torch.abs(Tt[-2] - Tb[-2]) <= delta_T) and (Tt[-2]+Tb[-2] >= best_T):
                    # episode is done, maybe refine? 
                    good_enough_flag = True
                    bonus = 7.0
                    print("Found good enough in eval run. Grats.")
                eval_reward = reward
                tot_eval_reward += eval_reward
                print(f"Eval reward: {eval_reward}")
                if good_enough_flag:
                    break
        
        # logging is now done here and with some different vals because I effed up earlier
        # --- Logging ---
        ewma_reward = 0.05 * episode_reward.item() + (1 - 0.05) * ewma_reward # done here instead
        writer.add_scalar("Episode Reward", episode_reward.item(), epoch)
        writer.add_scalar("EWMA", ewma_reward, epoch)
        writer.add_scalar("Eval/Episode Reward", tot_eval_reward, epoch)
        
        # saving the replay buffer after every epoch
        # buffer.save()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: reward={reward.item():.4f}  actor_loss={actor_loss.item():.4f}  critic_loss={critic_loss.item():.4f}")
            torch.save(actor.state_dict(), "models/actor_latest.pt")
            torch.save(critic.state_dict(), "models/critic_latest.pt")
            print(f"Saved models at epoch {epoch+1}.")
            
    writer.close()
    # try to save buffer here
    buffer.save()
    return actor, critic

if __name__ == "__main__":
    # load designs matrix and transform into graph representation using the imported function
    designs = np.load('FullDesign.npy').reshape(-1, 32, 32)
    # if you're going to unpad them, do so here
    designs = designs[:,1:31,9:23] # now shape should be (N, 30, 14)
    designs = designs[:, :, :7] # half, then we mirror to full
    graph_list = [create_graph_from_topology_matrix(d, use_8_conn=True, add_selfloops=False) for d in designs]
    print("Data loaded and turned to graph representration.")
    
    # Train (set to steps per epoch and batch size 1 to test that it works, then increase)
    actor, critic = train_ddpg(graph_list, epochs=300, steps_per_epoch=16, batch_size=16) # set to 2? for testing
    