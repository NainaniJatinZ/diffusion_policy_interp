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

def intervene_with_sae(layer_num, sae):
    def hook(module, input, output):
        recons, auxk, num_dead = sae(output)
        return recons
    return hook

layer_num = 4
handle = policy.model.decoder.layers[layer_num].register_forward_hook(intervene_with_sae(layer_num, sae))
for seed in seeds[2:]:
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
                    imgs.append(env.render(mode='rgb_array'))
                    step_idx += 1
                    pbar.update(1)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        print(f"Reward for {seed}: ", max(rewards))
                        break

    finally:
        # Remove hooks
        handle.remove()
        torch.cuda.empty_cache()
    break

# %%
from IPython.display import Video
vwrite('out/lowdim_recons.mp4', imgs)
Video('out/lowdim_recons.mp4', embed=True, width=256, height=256)


# %%

# define a random tensor of shape [1, 10, 256]
x = torch.randn(1, 10, 256)
recons, auxk, num_dead = sae(x)
recons.shape, auxk.shape, num_dead
# %%
recons.shape
# %%
