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
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import collections
import random

n_seeds = 20  # Number of random seeds to sweep
seed_threshold = 10000
obs_horizon = 2

# Generate n_seeds random seeds greater than 10000
seeds = [random.randint(seed_threshold + 1, 10 * seed_threshold) for _ in range(n_seeds)]

# Dictionary to collect activations for each layer across all seeds
all_layers_activations = {f"layer_{i}": [] for i in range(8)}

for seed in seeds:
    # Initialize environment and model for each seed
    env = PushTKeypointsEnv()
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()

    obs_deque = collections.deque([obs[:20]] * obs_horizon, maxlen=obs_horizon)
    done = False
    max_steps = 200
    step_idx = 0
    rewards = []
    policy.to('cuda')

    # Temporary storage for activations for this seed
    out_layers = {f"layer_{i}": [] for i in range(8)}

    # Function to create hooks for each layer
    def make_hook(layer_num):
        def hook(module, input, output):
            out_layers[f"layer_{layer_num}"].append(output)
        return hook

    # Register hooks for layers 0 through 7
    handles = []
    for i in range(8):
        handle = policy.model.decoder.layers[i].register_forward_hook(make_hook(i))
        handles.append(handle)

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
                    if step_idx > max_steps:
                        done = True
                    if done:
                        print(f"Reward for {seed}: ", max(rewards))
                        break
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
    
    # Reshape and collect activations for each layer for this seed
    for layer_name, activations in out_layers.items():
        concatenated_activations = torch.cat(activations, dim=0).view(-1, 256)  # Shape: [variable_length, 256]
        all_layers_activations[layer_name].append(concatenated_activations)
    print(concatenated_activations.shape)

# Concatenate activations across all seeds for each layer
for layer_name, activations_list in all_layers_activations.items():
    all_layers_activations[layer_name] = torch.cat(activations_list, dim=0)  # Shape: [total_activations, 256]

# Save the activations for each layer to a dictionary
torch.save(all_layers_activations, f'../data/all_layers_activations_{n_seeds}seeds.pt')
print("Saved activations for layers 0 to 7 across all seeds.")

# %%
n_seeds = 20
# load the activations back 
all_layers_activations = torch.load(f'../data/all_layers_activations_{n_seeds}seeds.pt')

# %%
