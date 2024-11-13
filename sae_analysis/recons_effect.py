# %%

import os 
os.chdir("../")
from sae.sae_model import SparseAutoencoder
import torch 

sae_path = "sae/results_layer4_dim2048_k64_auxk64_dead200/checkpoints/last.ckpt"

# load the weights 
sae_weights = torch.load(sae_path)

# %%
ckpt = {}
for k in sae_weights['state_dict'].keys():
    if k.startswith('sae_model.'):
        ckpt[k.split(".")[1]] = sae_weights['state_dict'][k]
ckpt.keys()

# %%
sae = SparseAutoencoder(256, 2048, 64, 64, 32, 200)
sae.load_state_dict(ckpt)
sae.to('cuda')

# %%
#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
import sys 
sys.path.append("../")
from helpers.pusht_helpers import *
import sys
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch


ckpt_path = "/work/pi_hzhang2_umass_edu/jnainani_umass_edu/Interp4Robotics/diffusionInterp/data/experiments/low_dim/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"


state_dict = torch.load(ckpt_path, map_location='cuda')
config = state_dict['cfg']
model_config = config['policy']['model']
model_config = {k: v for k, v in model_config.items() if not k.startswith('_target_')}
model = TransformerForDiffusion(**model_config)
noise_scheduler_config = config['policy']['noise_scheduler']
noise_scheduler = DDPMScheduler(**noise_scheduler_config)
# Extract other parameters needed for policy
policy_params = {
    'model': model,
    'noise_scheduler': noise_scheduler,
    'horizon': config['policy']['horizon'],
    'obs_dim': config['policy']['obs_dim'],
    'action_dim': config['policy']['action_dim'],
    'n_action_steps': config['policy']['n_action_steps'],
    'n_obs_steps': config['policy']['n_obs_steps'],
    'num_inference_steps': config['policy'].get('num_inference_steps', None),
    'obs_as_cond': config['policy'].get('obs_as_cond', False),
    'pred_action_steps_only': config['policy'].get('pred_action_steps_only', False),
}

# Instantiate the policy
policy = DiffusionTransformerLowdimPolicy(**policy_params)
# Load the model weights
policy.load_state_dict(state_dict['state_dicts']['model'])
# %%

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import collections
import random
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
policy.to('cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

n_seeds = 20  # Number of random seeds to sweep
seed_threshold = 10000
obs_horizon = 2

# Generate n_seeds random seeds greater than 10000
# seeds = [random.randint(seed_threshold + 1, 10 * seed_threshold) for _ in range(n_seeds)]
seeds = [78540,
 90318,
 40021,
 25426,
 50788,
 13783,
 37540,
 47567,
 10301,
 83703,
 95830,
 87380,
 41517,
 89203,
 33539,
 92410,
 30636,
 87390,
 52752,
 84098]

# Dictionary to collect activations for each layer across all seeds
# all_layers_activations = {f"layer_{i}": [] for i in range(8)}

# def intervene_with_sae(layer_num, sae):
#     def hook(module, input, output):
#         recons, auxk, num_dead = sae(output)
#         return recons
    # return hook
# inference_inputs = {}
layer_num = 4
# handle = policy.model.decoder.layers[layer_num].register_forward_hook(intervene_with_sae(layer_num, sae))
for seed in seeds:
    # Initialize environment and model for each seed
    env = PushTKeypointsEnv()
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    imgs = []  #
    obs_deque = collections.deque([obs[:20]] * obs_horizon, maxlen=obs_horizon)
    done = False
    max_steps = 200
    step_idx = 0
    rewards = []
    policy.to('cuda')
    inference_inputs = {}
    try: 
        with tqdm(total=max_steps, desc=f"Seed {seed} Eval") as pbar:
            while not done:
                B = 1
                obs_seq = np.stack(obs_deque)
                nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)

                with torch.no_grad():
                    # Normalization and action inference logic from your code
                    nobs = policy.normalizer['obs'].normalize(nobs)
                    B, _, Do = nobs.shape
                    To = policy.n_obs_steps
                    T = policy.horizon
                    Da = policy.action_dim
                    device = policy.device
                    dtype = policy.dtype
                    cond = nobs[:, :To]

                    shape = (B, T, Da)
                    if policy.pred_action_steps_only:
                        shape = (B, policy.n_action_steps, Da)
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                    generator = None
                    trajectory = torch.randn(
                        size=cond_data.shape, 
                        dtype=cond_data.dtype,
                        device=cond_data.device,
                        generator=generator)
            
                    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)

                    inference_inputs[step_idx] = []

                    for t in policy.noise_scheduler.timesteps:
                        trajectory[cond_mask] = cond_data[cond_mask]
                        step_input = {
                            "timestep": t,
                            "trajectory_input": trajectory.clone().cpu().numpy().tolist(),
                            "cond_input": cond.clone().cpu().numpy().tolist()
                        }
                        
                        # Append to this timestep's entry in the seed dictionary
                        inference_inputs[step_idx].append(step_input)

                        model_output = policy.model(trajectory, t, cond)

                        trajectory = policy.noise_scheduler.step(
                            model_output, t, trajectory, 
                            generator=generator,
                            **policy.kwargs
                        ).prev_sample

                    trajectory[cond_mask] = cond_data[cond_mask]
                    naction_pred = trajectory[..., :Da]
                    action_pred = policy.normalizer['action'].unnormalize(naction_pred)

                    start = To - 1
                    end = start + policy.n_action_steps
                    action = action_pred[:, start:end]

                naction = action.detach().to('cpu').numpy()
                action = naction[0]

                for i in range(len(action)):
                    obs, reward, done, _ = env.step(action[i])
                    obs_deque.append(obs[:20])
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))
                    step_idx += 1
                    pbar.update(1)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        print(f"Reward for {seed}: ", max(rewards))
                        break
        torch.save(inference_inputs, f"data/inference_inputs_seed_{seed}.pt")
    finally:
        # Remove hooks
        # handle.remove()
        torch.cuda.empty_cache()
    # break

# %%
from IPython.display import Video
vwrite('out/lowdim_recons.mp4', imgs)
Video('out/lowdim_recons.mp4', embed=True, width=256, height=256)

# %%
policy.normalizer['action'].unnormalize(torch.Tensor(inference_inputs[0][0]['trajectory_input']))

# %%
policy.normalizer['obs'].unnormalize(torch.Tensor(inference_inputs[0][0]['cond_input']))



# %%

# define a random tensor of shape [1, 10, 256]
x = torch.randn(1, 10, 256)
recons, auxk, num_dead = sae(x)
recons.shape, auxk.shape, num_dead
# %%
import torch
import pandas as pd
import os
from tqdm import tqdm

sae.to('cuda')
# Directory where inference input files are stored
data_dir = "data"
output_df = []

# List of features we want to monitor
feature_list = [i for i in range(0, 2047, 2)]

# Calculate total number of steps (files * step indices * timesteps)
seed_files = [f for f in os.listdir(data_dir) if f.endswith(".pt") and f.startswith("inference_inputs")][10:]
total_steps = 10 # len(seed_files)

# sum(
#     len(inference_inputs) * len(step_data)
#     for seed_file in seed_files
#     for inference_inputs in [torch.load(os.path.join(data_dir, seed_file))]
#     for step_data in inference_inputs.values()
# )

# Initialize a single progress bar
with tqdm(total=total_steps, desc="Processing All Seeds and Steps") as pbar:
    # Iterate over each saved .pt file in the directory
    for seed_file in seed_files:
        seed = int(seed_file.split('_')[-1].split('.')[0])  # Extract seed from filename
        inference_inputs = torch.load(os.path.join(data_dir, seed_file))
        
        # Iterate over each step_idx in the loaded data
        for step_idx, step_data in inference_inputs.items():
            # Each step contains multiple denoising steps, iterate through each timestep t
            for step_input in step_data:
                t = torch.Tensor([int(step_input["timestep"])]).to("cuda")
                trajectory_input = torch.tensor(step_input["trajectory_input"]).to("cuda", dtype=torch.float32)
                cond_input = torch.tensor(step_input["cond_input"]).to("cuda", dtype=torch.float32)
                layer_outs_data = {}
                
                def get_layer_out(layer_num):
                    def hook(module, input, output):
                        layer_outs_data[layer_num] = output.detach().cpu()
                    return hook

                layer_num = 4
                handle = policy.model.decoder.layers[layer_num].register_forward_hook(get_layer_out(layer_num))

                try:
                    with torch.no_grad():
                        model_output = policy.model(trajectory_input, t, cond_input)
                finally:
                    handle.remove()
                    torch.cuda.empty_cache()

                x = layer_outs_data[layer_num]
                x, mu, std = sae.LN(x.to('cuda'))
                x = x - sae.b_pre

                pre_acts = x @ sae.w_enc + sae.b_enc

                # latents: (BATCH_SIZE, D_EMBED, D_HIDDEN)
                latents = sae.topK_activation(pre_acts, k=sae.k)
                feature_acts = latents.flatten(0, 1)
                fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
                selected_acts = feature_acts[fired_mask][:, feature_list]

                # Log only non-zero activations in the data_row
                data_row = {
                    "seed": seed,
                    "step_idx": step_idx,
                    "timestep": t[0].detach().cpu().item(),
                    **{f"feature_{i}": selected_acts[:, idx].mean().item()
                       for idx, i in enumerate(feature_list) if (selected_acts[:, idx] > 0).any()}
                }
                output_df.append(data_row)
                
                # Update the progress bar after each timestep
        pbar.update(1)

# Convert output data to a DataFrame
activations_df = pd.DataFrame(output_df)
activations_df.head()

# Save to CSV
activations_df.to_csv("activations_summary_2.csv", index=False)


# %%

top_10_activations = activations_df.sort_values(
    f"feature_{36}", ascending=False
).head(10)

top_10_activations[['seed', 'step_idx', 'timestep', 'feature_36']]


# %%


import torch
import numpy as np
import collections
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Set the feature indices to analyze
feature_indices = np.random.choice([int(col.split('_')[-1]) for col in activations_df.columns if col.startswith("feature_")], 10, replace=False)
feature_indices
# %%
# Process each feature independently
for feature_idx in feature_indices:
    # Create a directory for this feature's visualizations
    os.makedirs(f"feature_viz/f{feature_idx}", exist_ok=True)
    
    # Find the top 10 activations for this feature and extract seeds and steps
    top_10_activations = activations_df.sort_values(f"feature_{feature_idx}", ascending=False).head(10)
    target_steps = top_10_activations[['seed', 'step_idx']].drop_duplicates()
    grouped_target_steps = target_steps.groupby('seed')['step_idx'].apply(list).to_dict()
    
    # Run the simulation for each seed for this specific feature
    for seed, target_step_idxs in grouped_target_steps.items():
        # Initialize environment and model for each seed
        env = PushTKeypointsEnv()
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        imgs = []
        obs_deque = collections.deque([obs[:20]] * obs_horizon, maxlen=obs_horizon)
        done = False
        max_steps = 200
        step_idx = 0
        rewards = []
        policy.to('cuda')

        try:
            with tqdm(total=max_steps, desc=f"Seed {seed} Eval for Feature {feature_idx}") as pbar:
                while not done:
                    B = 1
                    obs_seq = np.stack(obs_deque)
                    nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)

                    with torch.no_grad():
                        # Normalize observations and prepare for inference
                        nobs = policy.normalizer['obs'].normalize(nobs)
                        B, _, Do = nobs.shape
                        To = policy.n_obs_steps
                        T = policy.horizon
                        Da = policy.action_dim
                        device = policy.device
                        dtype = policy.dtype
                        cond = nobs[:, :To]

                        shape = (B, T, Da)
                        if policy.pred_action_steps_only:
                            shape = (B, policy.n_action_steps, Da)
                        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                        generator = None
                        trajectory = torch.randn(
                            size=cond_data.shape, 
                            dtype=cond_data.dtype,
                            device=cond_data.device,
                            generator=generator)
                
                        policy.noise_scheduler.set_timesteps(policy.num_inference_steps)

                        for t in policy.noise_scheduler.timesteps:
                            trajectory[cond_mask] = cond_data[cond_mask]
                            model_output = policy.model(trajectory, t, cond)

                            trajectory = policy.noise_scheduler.step(
                                model_output, t, trajectory, 
                                generator=generator,
                                **policy.kwargs
                            ).prev_sample

                        trajectory[cond_mask] = cond_data[cond_mask]
                        naction_pred = trajectory[..., :Da]
                        action_pred = policy.normalizer['action'].unnormalize(naction_pred)

                        start = To - 1
                        end = start + policy.n_action_steps
                        action = action_pred[:, start:end]

                    naction = action.detach().to('cpu').numpy()
                    action = naction[0]

                    for i in range(len(action)):
                        obs, reward, done, _ = env.step(action[i])
                        obs_deque.append(obs[:20])
                        rewards.append(reward)
                        step_idx += 1
                        pbar.update(1)

                        # Capture images if the current step_idx is in the target list for this feature
                        if step_idx in target_step_idxs:
                            img = env.render(mode='rgb_array')
                            img_pil = Image.fromarray(img)
                            # Save the image for the specific seed and step_idx under this feature's folder
                            img_filename = f"feature_viz/f{feature_idx}/seed_{seed}_step_{step_idx}.png"
                            img_pil.save(img_filename)

                        if step_idx > max_steps or done:
                            done = True
                            print(f"Reward for {seed}: ", max(rewards))
                            break
        finally:
            # Ensure resources are cleared
            torch.cuda.empty_cache()

# %%
            
import torch
import numpy as np
import collections
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Load activations_summary.csv
activations_summary = pd.read_csv("activations_summary.csv")

# Define the target features
target_features = feature_indices

# Create directories for each feature to save images
for feature_idx in target_features:
    os.makedirs(f"feature_viz/f{feature_idx}", exist_ok=True)

# Process each feature in the target_features list
for feature_idx in target_features:
    # Find the top 10 activations for the current feature
    top_10_activations = activations_summary.sort_values(f"feature_{feature_idx}", ascending=False).head(10)
    target_steps = top_10_activations[['seed', 'step_idx']].drop_duplicates()
    grouped_target_steps = target_steps.groupby('seed')['step_idx'].apply(list).to_dict()
    
    # Run the simulation for each unique seed for this feature
    for seed, target_step_idxs in grouped_target_steps.items():
        # Initialize environment and model for the seed
        env = PushTKeypointsEnv()
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        imgs = []
        obs_deque = collections.deque([obs[:20]] * obs_horizon, maxlen=obs_horizon)
        done = False
        max_steps = 200
        step_idx = 0
        rewards = []
        policy.to('cuda')

        try:
            with tqdm(total=max_steps, desc=f"Seed {seed} Eval for Feature {feature_idx}") as pbar:
                while not done:
                    B = 1
                    obs_seq = np.stack(obs_deque)
                    nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)

                    with torch.no_grad():
                        # Normalize observations and prepare for inference
                        nobs = policy.normalizer['obs'].normalize(nobs)
                        B, _, Do = nobs.shape
                        To = policy.n_obs_steps
                        T = policy.horizon
                        Da = policy.action_dim
                        device = policy.device
                        dtype = policy.dtype
                        cond = nobs[:, :To]

                        shape = (B, T, Da)
                        if policy.pred_action_steps_only:
                            shape = (B, policy.n_action_steps, Da)
                        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                        generator = None
                        trajectory = torch.randn(
                            size=cond_data.shape, 
                            dtype=cond_data.dtype,
                            device=cond_data.device,
                            generator=generator)
                
                        policy.noise_scheduler.set_timesteps(policy.num_inference_steps)

                        for t in policy.noise_scheduler.timesteps:
                            trajectory[cond_mask] = cond_data[cond_mask]
                            model_output = policy.model(trajectory, t, cond)

                            trajectory = policy.noise_scheduler.step(
                                model_output, t, trajectory, 
                                generator=generator,
                                **policy.kwargs
                            ).prev_sample

                        trajectory[cond_mask] = cond_data[cond_mask]
                        naction_pred = trajectory[..., :Da]
                        action_pred = policy.normalizer['action'].unnormalize(naction_pred)

                        start = To - 1
                        end = start + policy.n_action_steps
                        action = action_pred[:, start:end]

                    naction = action.detach().to('cpu').numpy()
                    action = naction[0]

                    for i in range(len(action)):
                        obs, reward, done, _ = env.step(action[i])
                        obs_deque.append(obs[:20])
                        rewards.append(reward)
                        step_idx += 1
                        pbar.update(1)

                        # Capture images if the current step_idx is in the target list for this feature
                        if step_idx in target_step_idxs:
                            img = env.render(mode='rgb_array')
                            img_pil = Image.fromarray(img)
                            # Save the image for this specific seed and step_idx under the feature's folder
                            img_filename = f"feature_viz/f{feature_idx}/seed_{seed}_step_{step_idx}.png"
                            img_pil.save(img_filename)

                        if step_idx > max_steps or done:
                            done = True
                            print(f"Reward for {seed}: ", max(rewards))
                            break
        finally:
            # Ensure resources are cleared
            torch.cuda.empty_cache()



# %%


import torch
import numpy as np
import collections
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image 
feature_idx = 36
# Load top 10 activations data
top_10_activations = activations_df.sort_values(f"feature_{feature_idx}", ascending=True).head(10)
target_steps = top_10_activations[['seed', 'step_idx']].drop_duplicates()

# Directory to save images
os.makedirs(f"feature_viz/f{feature_idx}", exist_ok=True)

for _, row in target_steps.iterrows():
    seed = int(row['seed'])
    target_step_idx = int(row['step_idx'])
    
    # Initialize environment and model for each seed
    env = PushTKeypointsEnv()
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    imgs = []
    obs_deque = collections.deque([obs[:20]] * obs_horizon, maxlen=obs_horizon)
    done = False
    max_steps = 200
    step_idx = 0
    rewards = []
    policy.to('cuda')

    try:
        with tqdm(total=max_steps, desc=f"Seed {seed} Eval") as pbar:
            while not done:
                B = 1
                obs_seq = np.stack(obs_deque)
                nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)

                with torch.no_grad():
                    # Normalize observations and prepare for inference
                    nobs = policy.normalizer['obs'].normalize(nobs)
                    B, _, Do = nobs.shape
                    To = policy.n_obs_steps
                    T = policy.horizon
                    Da = policy.action_dim
                    device = policy.device
                    dtype = policy.dtype
                    cond = nobs[:, :To]

                    shape = (B, T, Da)
                    if policy.pred_action_steps_only:
                        shape = (B, policy.n_action_steps, Da)
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                    generator = None
                    trajectory = torch.randn(
                        size=cond_data.shape, 
                        dtype=cond_data.dtype,
                        device=cond_data.device,
                        generator=generator)
            
                    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)

                    for t in policy.noise_scheduler.timesteps:
                        trajectory[cond_mask] = cond_data[cond_mask]
                        model_output = policy.model(trajectory, t, cond)

                        trajectory = policy.noise_scheduler.step(
                            model_output, t, trajectory, 
                            generator=generator,
                            **policy.kwargs
                        ).prev_sample

                    trajectory[cond_mask] = cond_data[cond_mask]
                    naction_pred = trajectory[..., :Da]
                    action_pred = policy.normalizer['action'].unnormalize(naction_pred)

                    start = To - 1
                    end = start + policy.n_action_steps
                    action = action_pred[:, start:end]

                naction = action.detach().to('cpu').numpy()
                action = naction[0]

                for i in range(len(action)):
                    obs, reward, done, _ = env.step(action[i])
                    obs_deque.append(obs[:20])
                    rewards.append(reward)
                    step_idx += 1
                    pbar.update(1)

                    # Capture image if step matches target
                    if step_idx == target_step_idx:
                        img = env.render(mode='rgb_array')
                        imgs.append(img)
                        # Save the image for the specific seed and step_idx
                        img_pil = Image.fromarray(img)
                        img_filename = f"feature_viz/f{feature_idx}/seed_{seed}_step_{step_idx}.png"
                        img_pil.save(img_filename)

                    if step_idx > max_steps or done:
                        done = True
                        print(f"Reward for {seed}: ", max(rewards))
                        break
    finally:
        # Ensure resources are cleared
        torch.cuda.empty_cache()




# %%
#                 # Dictionary to store layer outputs for this step
#                 layer_outs_data = {}

#                 # Hook to capture activations at the desired layer
#                 def get_layer_out(layer_num):
#                     def hook(module, input, output):
#                         layer_outs_data[layer_num] = output.detach().cpu()
#                     return hook

#                 layer_num = 30
#                 handle = model.transformer.blocks[layer_num].register_forward_hook(get_layer_out(layer_num))

#                 try:
#                     # Run the model with trajectory and condition inputs
#                     with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                         model(trajectory_input, cond_input)

#                 finally:
#                     handle.remove()
#                     torch.cuda.empty_cache()

#                 # Encode the layer output using the autoencoder
#                 feature_acts = ae.encode(layer_outs_data[layer_num])

#                 # Flatten and filter for features of interest
#                 feature_acts = feature_acts.flatten(0, 1)
#                 selected_acts = feature_acts[:, feature_list]

#                 # Add the data to the DataFrame
#                 data_row = {
#                     "seed": seed,
#                     "step_idx": step_idx,
#                     "timestep": t,
#                     **{f"feature_{i}": selected_acts[:, i].mean().item() for i in range(selected_acts.shape[1])}
#                 }
#                 output_df.append(data_row)

# # Convert the list of dictionaries to a DataFrame
# activations_df = pd.DataFrame(output_df)

# # Save the DataFrame to a CSV or directly return for further analysis
# activations_df.to_csv("activations_summary.csv", index=False)

# %%
