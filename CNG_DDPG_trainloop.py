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
import pathlib
import json
import uuid
import datetime
import matplotlib.pyplot as plt
import pickle 
import traceback

# imports from other files for RL and graph
from RL_help_funcs import RewardHelper, PriorityReplayBuffer, OUNoise, noise_scale_scheduler
from action_to_designs import decode_actions_to_design, topology_matrix_from_decoded_actions, mirror_design
from GCN_ActorCritic import EdgeAwareActor, EdgeAwareCritic
from turnToGraph import create_graph_from_topology_matrix
from glabados import DesignCache, simulation_cacher

# import from other files for the MEEP simulation
from simulation_funcs import run_meep_sim_wsl 

# tensorboard
from torch.utils.tensorboard import SummaryWriter

import argparse


def _ensure_saved_designs_dir():
    p = pathlib.Path("SavedDesigns")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_good_design(hole_flag_full, diameters_full, Tt, Tb, epoch, step, reward, actor=None, critic=None, tag="train"):
    """Persist a found good design: numpy arrays, PNG, metadata and optional model checkpoints.

    Files go into SavedDesigns/ with a unique timestamped filename.
    """
    out_dir = _ensure_saved_designs_dir()
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:8]
    try:
        total_T = float(Tt[-2] + Tb[-2]) if hasattr(Tt, '__len__') else float(Tt + Tb)
    except Exception:
        total_T = float(0.0)
    fname_base = f"good_{tag}_ep{epoch:04d}_st{step:03d}_T{total_T:.4f}_{ts}_{uid}"

    # topology matrix
    try:
        from action_to_designs import topology_matrix_from_decoded_actions
        topo = topology_matrix_from_decoded_actions(hole_flag_full, diameters_full)
    except Exception:
        topo = None

    # Ensure we are saving plain numpy arrays (in case tensors were passed)
    try:
        hole_flag_np = np.asarray(hole_flag_full)
        diameters_np = np.asarray(diameters_full)
    except Exception:
        hole_flag_np = None
        diameters_np = None

    # Save arrays and metadata (with safer error reporting)
    meta = {
        "epoch": int(epoch),
        "step": int(step),
        "timestamp_utc": ts,
        "uid": uid,
        "total_T": float(total_T),
        "Tt_sample": list(map(float, Tt)) if hasattr(Tt, '__iter__') else [float(Tt)],
        "Tb_sample": list(map(float, Tb)) if hasattr(Tb, '__iter__') else [float(Tb)],
        "reward": float(reward),
    }

    def _log_save_error(stage, exc_text):
        # write a small error log next to saved files and print to stdout
        log_path = out_dir / (fname_base + f".{stage}.error.log")
        try:
            with open(log_path, 'w') as lf:
                lf.write(exc_text)
        except Exception:
            # if this fails, at least print
            print(f"Failed to write error log to {log_path}")
        print(f"[save_good_design] {stage} error:\n{exc_text}")

    npz_path = out_dir / (fname_base + ".npz")
    try:
        # Save only the sanitized numpy arrays and topology/metadata
        np.savez_compressed(
            npz_path,
            hole_flag=hole_flag_np,
            diameters=diameters_np,
            topology=topo,
            metadata=meta,
        )
        # quick sanity-check: ensure file exists and is not empty
        if not npz_path.exists() or npz_path.stat().st_size == 0:
            raise IOError(f"Saved .npz missing or empty: {npz_path}")
    except Exception:
        exc_text = traceback.format_exc()
        _log_save_error('npz_save', exc_text)

    # save PNG visualization (diameters heatmap)
    try:
        plt.figure(figsize=(4, 6))
        if topo is not None:
            plt.imshow(topo, cmap='viridis')
        else:
            # fallback: reshape diameters to some reasonable shape if possible
            arr = np.asarray(diameters_np) if diameters_np is not None else None
            if arr is None:
                raise ValueError("No diameters available to create PNG visualization")
            if arr.ndim == 1:
                # try to infer shape: if length divisible by 14 or 30
                if arr.size % 14 == 0:
                    arr2 = arr.reshape((-1, 14))
                elif arr.size % 30 == 0:
                    arr2 = arr.reshape((30, -1))
                else:
                    # fallback to square-ish
                    s = int(np.sqrt(arr.size))
                    if s * s == arr.size:
                        arr2 = arr.reshape((s, s))
                    else:
                        arr2 = arr.reshape((arr.size, 1))
            else:
                arr2 = arr
            plt.imshow(arr2, cmap='viridis')
        plt.colorbar()
        plt.title(f"{fname_base}")
        plt.tight_layout()
        png_path = out_dir / (fname_base + ".png")
        plt.savefig(png_path)
        plt.close()
    except Exception:
        exc_text = traceback.format_exc()
        _log_save_error('png_save', exc_text)

    # save metadata JSON
    try:
        with open(out_dir / (fname_base + ".json"), 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        exc_text = traceback.format_exc()
        _log_save_error('json_save', exc_text)

    # optionally save model weights for reference
    try:
        if actor is not None:
            torch.save(actor.state_dict(), out_dir / (fname_base + ".actor.pt"))
        if critic is not None:
            torch.save(critic.state_dict(), out_dir / (fname_base + ".critic.pt"))
    except Exception:
        exc_text = traceback.format_exc()
        _log_save_error('model_save', exc_text)


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
               eval_interval=1,
               binary_matrix_shape=(30,14)
               ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    initial_graph = random.choice(graph_list) # randomly select an initial design from list 
    
    num_nodes, num_features = initial_graph.x.size(0), initial_graph.x.size(1)

    action_dim = (num_nodes,num_features) # action dimension same as node features (hole flag and diameter per node)
    
    # do it here once
    half_binary_matrix_shape = (binary_matrix_shape[0], binary_matrix_shape[1]//2)

    actor = EdgeAwareActor(in_channels=initial_graph.x.size(1), 
                           hidden_dim=128, 
                           out_channels=2, 
                           edge_dim=initial_graph.edge_attr.size(1), num_layers=5).to(device)
    critic = EdgeAwareCritic(in_channels=initial_graph.x.size(1), 
                             action_dim=2, hidden_dim=128, 
                             edge_dim=initial_graph.edge_attr.size(1)).to(device)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    optim_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    
    # exploration noise
    noise = OUNoise(shape=(action_dim[0],2), mu=0.0, theta=0.15, sigma=0.2, scale=2.0)# good results so far
    buffer = PriorityReplayBuffer(buffer_capacity, initial_mode='td_error')
    
    # tensorboard's writer defined here
    writer = SummaryWriter(log_dir='runs/ddpg_training/mini_1x2_2')
    
    # global steps or total number of steps
    total_steps = 0 # count environment interactions
    
    # ewma_reward, for keeping track of a trend and therefore whether or not this shite is learning
    ewma_reward = 0
    
    # init cache object and its directory
    cache = DesignCache('./CacheLocation/cache_mini.json') # when designs change format, change cache...
    
    # reward helper to keep track of some params
    reward_helper = RewardHelper(template="1x2", target_ratio=1.5) # need to make the template passable arg from main

    for epoch in range(epochs):
        # some things need resetting per episode
        episode_reward = 0
        episode_length = 0 # should end episode once we have a nice enough transmission profile, no? how many steps did it take to arrive there?
        next_graph = random.choice(graph_list) # don't want to always start from same design. P_0
        noise.reset()
        noise.scale = noise_scale_scheduler(epoch,
                                            initial_scale=2.0, decay_rate=0.5,
                                            min_scale=0.05) # best test some jazz
        
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
            action = torch.clamp(action, 0, 1)

            # ── Decode & simulate ──────────────────────
            hole_flag, diameters = decode_actions_to_design(action) # equivalent to next state, with T
            hole_flag_full, diameters_full = mirror_design(hole_flag, diameters, design_shape=binary_matrix_shape) # mirroring if halved
            
            # keep the fully worked action that gets used in MEEP, but halved because half
            executed_action = torch.stack([
                torch.tensor(hole_flag_full.reshape(binary_matrix_shape)[:,:half_binary_matrix_shape[1]].reshape(-1), dtype=torch.float, device=device),
                torch.tensor(diameters_full.reshape(binary_matrix_shape)[:,:half_binary_matrix_shape[1]].reshape(-1), dtype=torch.float, device=device)
                ], dim=1) # (2,30,7) -> flatten with reshape(-1), so we have (N,2)
            Tt, Tb = simulation_cacher(cache, hole_flag_full, diameters_full) # they are returned as lists at this point
            # make them into tensors or next function fails
            Tt = torch.Tensor(Tt)
            Tb = torch.Tensor(Tb)

            # TODO: use the reward helper to obtain the reward
            reward, good_enough = reward_helper.checking_criteria(Tt, Tb)

            # if next_graph = graph, it'd be stateless, which it's not, so create graph from new design after action
            updates_design = topology_matrix_from_decoded_actions(hole_flag_full, diameters_full, shape=binary_matrix_shape)
            # we have to create the graph representation of the next design, doughnut, same params
            next_graph = create_graph_from_topology_matrix(updates_design[:,:half_binary_matrix_shape[1]], True, True).to(device)

            # ───── check if episode termination criteria is met ─────
            if good_enough:
                reward_helper.update_best_params(Tt, Tb)
                try:
                    save_good_design(hole_flag_full, diameters_full, Tt, Tb, epoch, step, reward, actor=actor, critic=critic, tag="train")
                except Exception:
                    print("Error saving good design.")

            # ───── update episode length and running rewards ────
            episode_length += 1
            episode_reward += reward

            # ── Store in buffer ─────────────────────────
            buffer.push(graph, executed_action, reward, next_graph, good_enough)
            
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
                
            if good_enough:
                # print summary and end episode; saving already occurs at detection time above
                print(f"Episode Length: {episode_length}. Good enough: ({Tt},{Tb}), R:{reward}")
                break
        
        # pure evaluation
        next_graph = initial_graph.to(device) # start each evaluation form the same initial design
        if epoch % eval_interval == 0:
            tot_eval_reward = 0 
            eval_graph = next_graph
            
            #for step in range(steps_per_epoch):
            actor.eval()
            with torch.no_grad():
                action = actor(eval_graph.x, eval_graph.edge_index, eval_graph.edge_attr)
            action = torch.clamp(action, 0, 1)
            
            hole_flag, diameters = decode_actions_to_design(action) # equivalent to next state, with T
            hole_flag_full, diameters_full = mirror_design(hole_flag, diameters, design_shape=binary_matrix_shape) # mirroring if halved
            Tt, Tb = simulation_cacher(cache, hole_flag_full, diameters_full) # they are returned as lists at this point
            # make them into tensors or next function fails
            Tt = torch.Tensor(Tt)
            Tb = torch.Tensor(Tb)

            # reward should come from reward helper like in training 
            reward, good_enough = reward_helper.checking_criteria(Tt, Tb)

            # if next_graph = graph, it'd be stateless, which it's not, so create graph from new design after action
            updates_design = topology_matrix_from_decoded_actions(hole_flag_full, diameters_full, binary_matrix_shape)
            # we have to create the graph representation of the next design, doughnut, same params
            next_graph = create_graph_from_topology_matrix(updates_design[:,:half_binary_matrix_shape[1]], True, True).to(device)
            if good_enough:
                print("Found good enough in eval run. Grats.")
                try:
                    save_good_design(hole_flag_full, diameters_full, Tt, Tb, epoch, step, reward, actor=actor, critic=critic, tag="eval")
                except Exception:
                    print("Error saving good design from eval.")
            eval_reward = reward
            tot_eval_reward += eval_reward
            print(f"Eval reward: {eval_reward}")
            #if good_enough:
            #    break
        
        # logging is now done here and with some different vals because I effed up earlier
        # --- Logging ---
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward # done here instead
        writer.add_scalar("Episode Reward", episode_reward, epoch)
        writer.add_scalar("EWMA", ewma_reward, epoch)
        writer.add_scalar("Eval/Episode Reward", tot_eval_reward, epoch)
        
        # saving the replay buffer after every epoch
        # buffer.save()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: reward={reward:.4f}  actor_loss={actor_loss.item():.4f}  critic_loss={critic_loss.item():.4f}")
            torch.save(actor.state_dict(), "models/actor_latest.pt")
            torch.save(critic.state_dict(), "models/critic_latest.pt")
            print(f"Saved models at epoch {epoch+1}.")
            
    writer.close()
    # try to save buffer here
    buffer.save()
    return actor, critic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_designs', help="Load design binary matrices from .npy files.", action="store_true")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--steps_per_epoch', type=int, default=16, help="Steps per epoch.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--binary_shape', type=str, help="Shape of the binary design matrices, e.g., '32,32'.")

    args = parser.parse_args()
    # TODO: have to pass the arguments to the train loop and check hard coded matrix shapes
    if args.load_designs:
        print("Loading designs from .npy files...")

        # load designs matrix and transform into graph representation using the imported function
        designs = np.load('FullDesign.npy').reshape(-1, 32, 32)
        # if you're going to unpad them, do so here
        designs = designs[:,1:31,9:23] # now shape should be (N, 30, 14)
        designs = designs[:, :, :7] # half, then we mirror to full
        graph_list = [create_graph_from_topology_matrix(d, use_8_conn=True, add_selfloops=False) for d in designs]
        print("Data loaded and turned to graph representration.")

    else:
        shape_parts = args.binary_shape.split(',')
        if len(shape_parts) != 2:
            raise ValueError("binary_shape must be in the format 'rows,cols', e.g., '32,32'.")
        rows, cols = map(int, shape_parts)
        print(f"Using binary shape: {rows}x{cols}")

        print("Using base template design only and converting to graph representation...")
        # template design binary matrix will only have 0s only (since no etched holes)
        design = np.zeros((rows, cols), dtype=np.int8)
        design = design[:, :cols//2] # half, then we mirror to full
        graph_list = [create_graph_from_topology_matrix(design, use_8_conn=True, add_selfloops=False)]
        print("Template design graph representation created.")
    
    # Train (set to steps per epoch and batch size 1 to test that it works, then increase)
    actor, critic = train_ddpg(graph_list, 
                               epochs=args.epochs,
                               steps_per_epoch=args.steps_per_epoch, 
                               batch_size=args.batch_size,
                               binary_matrix_shape=(rows, cols)
                               )
