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

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

import torch
import torch.nn.functional as F
import math

# A list to store attention patterns for later analysis
multi_attention_patterns = []
self_attention_patterns = []
action_entropy_self = []
action_entropy_multi = []

def compute_token_wise_entropy(attention_patterns):
    """Compute token-wise entropy over time for each head, averaged over time."""
    entropy_over_time = []

    for attention_weights in attention_patterns:
        # Normalize and compute entropy
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        entropies = -torch.sum(attention_weights * torch.log(attention_weights + 1e-6), dim=-1)
        entropy_over_time.append(entropies)

    entropy_over_time = torch.stack(entropy_over_time)  # Shape: [time_steps, batch_size, num_heads, seq_length]
    avg_token_entropy = entropy_over_time.mean(dim=0)   # Average over time dimension
    return avg_token_entropy.squeeze(0)  # Remove batch dimension


def multi_attention_hook(module, input, output):
    # Unpack the inputs
    query, key, value = input[0], input[1], input[2]
    
    # Get the projection weights and biases
    in_proj_weight = module.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
    in_proj_bias = module.in_proj_bias      # Shape: (3*embed_dim)
    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    # Split the projection weights and biases for query, key, and value
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
    
    # Apply linear transformations for query, key, and value
    q = F.linear(query, q_proj_weight, q_proj_bias)  # Shape: (batch_size, seq_length, embed_dim)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)

    # Reshape and transpose to separate heads
    batch_size, seq_length, _ = q.size()
    _, enc_seq_length, _ = k.size()
    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)

    # Calculate scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Store the attention patterns for later inspection
    multi_attention_patterns.append(attn_weights.detach().cpu())

def self_attention_hook(module, input, output):
    # Unpack the inputs
    query, key, value = input[0], input[1], input[2]
    
    # Get the projection weights and biases
    in_proj_weight = module.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
    in_proj_bias = module.in_proj_bias      # Shape: (3*embed_dim)
    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    # Split the projection weights and biases for query, key, and value
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
    
    # Apply linear transformations for query, key, and value
    q = F.linear(query, q_proj_weight, q_proj_bias)  # Shape: (batch_size, seq_length, embed_dim)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)

    # Reshape and transpose to separate heads
    batch_size, seq_length, _ = q.size()
    _, enc_seq_length, _ = k.size()
    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)

    # Calculate scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Store the attention patterns for later inspection
    self_attention_patterns.append(attn_weights.detach().cpu())

policy.to('cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Run the environment simulation twice with different seeds
seeds = [14309903, 1000578]  # Example of two seeds
seeds = [1000, 574823]
action_entropy_multi_seeds = []
action_entropy_self_seeds = []
scores = []
all_obs = {}  # Dictionary to store observations by seed
all_imgs = {}  # Dictionary to store images by seed

for seed in seeds:
    # Set seed for reproducibility
    env = PushTKeypointsEnv()
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    obs_horizon = 2
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs[:20]] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    max_steps = 200
    step_idx = 0
    # Initialize entropy lists and rewards
    action_entropy_multi = []
    action_entropy_self = []
    rewards = []
    obs_list = [obs[:20]]  # To store observations for this seed
    imgs = []  # To store images for this seed
    multi_attn_module = policy.model.decoder.layers[7].multihead_attn
    handle_multi = multi_attn_module.register_forward_hook(multi_attention_hook)

    self_attn_module = policy.model.decoder.layers[7].self_attn
    handle_self = self_attn_module.register_forward_hook(self_attention_hook)
    multi_attention_patterns = []
    self_attention_patterns = []

    try: 
        with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
            while not done:
                B = 1
                obs_seq = np.stack(obs_deque)
                # infer action
                nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)

                with torch.no_grad():
                    nobs = policy.normalizer['obs'].normalize(nobs)
                    B, _, Do = nobs.shape
                    To = policy.n_obs_steps
                    T = policy.horizon
                    Da = policy.action_dim

                    # build input
                    device = policy.device
                    dtype = policy.dtype

                    # handle different ways of passing observation
                    cond = None
                    cond_data = None
                    cond_mask = None
                    cond = nobs[:,:To]

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
            
                    # set step values
                    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)

                    for t in policy.noise_scheduler.timesteps:
                        # 1. apply conditioning
                        trajectory[cond_mask] = cond_data[cond_mask]

                        # 2. predict model output
                        model_output = model(trajectory, t, cond)

                        # 3. compute previous image: x_t -> x_t-1
                        trajectory = policy.noise_scheduler.step(
                            model_output, t, trajectory, 
                            generator=generator,
                            **policy.kwargs
                            ).prev_sample
                    
                    # finally make sure conditioning is enforced
                    trajectory[cond_mask] = cond_data[cond_mask] 
                    multi_entropy = compute_token_wise_entropy(multi_attention_patterns)
                    self_entropy = compute_token_wise_entropy(self_attention_patterns)
                    print(len(multi_attention_patterns))
                    
                    # unnormalize prediction
                    naction_pred = trajectory[...,:Da]
                    action_pred = policy.normalizer['action'].unnormalize(naction_pred)

                    start = To - 1
                    end = start + policy.n_action_steps
                    action = action_pred[:,start:end]
                    # Only keep relevant tokens for action steps
                    action_entropy_multi.append(multi_entropy[:, start:end])
                    action_entropy_self.append(self_entropy[:, start:end])
                    # Clear the lists for next sequence
                    multi_attention_patterns.clear()
                    self_attention_patterns.clear()
                    print(len(multi_attention_patterns))

                naction = action.detach().to('cpu').numpy()
                # print(naction.shape)
                action = naction[0]

                # break
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _ = env.step(action[i])
                    # save observations
                    # print(obs)
                    obs_deque.append(obs[:20])
                    obs_list.append(obs[:20])
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
    finally:
        handle_multi.remove()
        handle_self.remove()
        torch.cuda.empty_cache()
    
    action_entropy_multi_seeds.append(torch.stack(action_entropy_multi, dim=0))
    action_entropy_self_seeds.append(torch.stack(action_entropy_self, dim=0))
    scores.append(max(rewards))
    # Save observations and images for the current seed
    all_obs[seed] = obs_list
    all_imgs[seed] = imgs

    print('Score: ', max(rewards))

# %%
import matplotlib.pyplot as plt
# Plotting function to compare entropies from different seeds and include scores in the title
# Function to reshape and plot entropies with multi-seed
# Function to reshape and plot entropies with multi-seed, with each head in a separate plot
def plot_entropy_over_actions_multi_seed(entropies_seeds, attention_type, heads, num_actions, seeds, scores):
    """Reshape entropy values and plot each head in a separate plot with two curves for each seed."""
    for head in range(heads):
        plt.figure(figsize=(10, 5))
        
        # Plot the entropy for each seed on the same plot for the current head
        for i, seed_entropy in enumerate(entropies_seeds):
            # Reshape from [26, 4, 8] to [208, 4]
            reshaped_entropy = seed_entropy.permute(1, 0, 2).reshape(heads, -1).T  # Shape: [208, 4]
            
            # Plot the specific head's entropy for the current seed
            plt.plot(
                range(num_actions[i]),
                reshaped_entropy[:, head].cpu().numpy(),
                label=f'Seed {seeds[i]} (Score: {scores[i]})'
            )
        
        # Customize each plot
        plt.title(f'{attention_type} Attention Entropy for Head {head+1} Over Actions')
        plt.xlabel('Action Step')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)
        plt.show()

# Determine the number of heads and action steps based on your data shape
heads = 4  # Assuming we have 4 heads
num_actions = [176, 208]  # After reshaping from [26, 4, 8] to [208, 4]

# Plot reshaped Multihead Attention Entropy with both seeds and scores
plot_entropy_over_actions_multi_seed(action_entropy_multi_seeds, "Multihead", heads, num_actions, seeds, scores)

# Plot reshaped Self Attention Entropy with both seeds and scores
plot_entropy_over_actions_multi_seed(action_entropy_self_seeds, "Self", heads, num_actions, seeds, scores)

# %%

action_entropy_multi = torch.stack(action_entropy_multi, dim=0)  # Shape: [num_actions, num_heads, seq_length]
action_entropy_self = torch.stack(action_entropy_self, dim=0)

from IPython.display import Video
vwrite('lowdim1.mp4', imgs)
Video('lowdim1.mp4', embed=True, width=256, height=256)

# %%

def plot_entropy_over_actions(entropies, attention_type, heads, num_actions):
    """Plot entropy variation over actions for each head."""

    for head in range(heads):
        reshaped_entropy = entropies.permute(1, 0, 2).reshape(heads, -1).T 
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_actions), reshaped_entropy[:, head], label=f'Head {head+1}')
        plt.title(f'{attention_type} Attention Entropy for Head {head+1} Over Actions')
        plt.xlabel('Action Step')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)
        plt.show()

plot_entropy_over_actions(action_entropy_self_seeds[-1], "Self", 4, 144)
# plot_entropy_over_actions(action_entropy_self_seeds, "Self", heads, num_actions, seeds, scores)

# %%



# %%

import matplotlib.pyplot as plt
import numpy as np

# Set up colors for each keypoint index (1 through 9)
colors = plt.cm.tab10(np.linspace(0, 1, 9))  # 9 distinct colors for keypoints 1 through 9

def plot_entropy_with_keypoints(entropies, observations, attention_type, heads, num_actions):
    """Plot entropy variation over actions for each head, with colored points based on closest keypoint."""
    
    for head in range(heads):
        reshaped_entropy = entropies.permute(1, 0, 2).reshape(heads, -1).T  # Shape [num_actions, heads]
        reshaped_entropy = reshaped_entropy[:num_actions]
        plt.figure(figsize=(10, 5))
        
        for action_idx in range(num_actions):
            # Get the observation corresponding to the current action
            obs = observations[action_idx]
            
            # Extract x and y coordinates of the agent and keypoints
            x_coords = obs[0:20:2]
            y_coords = obs[1:20:2]
            
            agent_x, agent_y = x_coords[-1], y_coords[-1]  # Agent's position
            # print(obs)
            # Compute distances to keypoints 1 through 9
            distances = [np.sqrt((agent_x/512 - x_coords[j]/512)**2 + (agent_y/512 - y_coords[j]/512)**2) for j in range(0, 9)]
            # print(distances)
            closest_keypoint_idx = np.argmin(distances) + 1  # Closest keypoint index (1-based)
            
            # Plot entropy for the current action and head with color based on closest keypoint
            plt.scatter(action_idx, reshaped_entropy[action_idx, head], color=colors[closest_keypoint_idx - 1],
                        label=f'Keypoint {closest_keypoint_idx}' if action_idx == 0 else "")
            
            # Annotate the closest keypoint index
            plt.text(action_idx, reshaped_entropy[action_idx, head], str(closest_keypoint_idx),
                     ha='center', va='bottom', fontsize=8)
        
        # Title and labels
        plt.title(f'{attention_type} Attention Entropy for Head {head+1} Over Actions')
        plt.xlabel('Action Step')
        plt.ylabel('Entropy')
        
        # Only show legend once per keypoint index
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True)
        plt.show()

# Call the function for the last seed's self-attention entropies
plot_entropy_with_keypoints(action_entropy_self_seeds[-1], all_obs[seeds[-1]], "Self", 4, 142)

# %%

# Flatten the entropy tensor to [num_actions, heads] and then average across heads
reshaped_entropy = action_entropy_self_seeds[-1].permute(1, 0, 2).reshape(4, -1).T  # Shape [142, 4]
mean_entropy_per_action = reshaped_entropy.mean(dim=1)  # Average entropy across heads for each action

# Get indices of actions with the lowest and highest entropy values
lowest_entropy_indices = torch.topk(mean_entropy_per_action, 3, largest=False).indices
highest_entropy_indices = torch.topk(mean_entropy_per_action, 3, largest=True).indices
# highest_entropy_index = torch.argmax(mean_entropy_per_action)

# Convert indices to lists for easy viewing
lowest_entropy_indices = lowest_entropy_indices.tolist()
highest_entropy_indices = highest_entropy_indices.tolist()

# Output indices
print("Top 3 actions with lowest entropy indices:", lowest_entropy_indices)
print("Top 3 actions with highest entropy indices:", highest_entropy_indices)

# %%
import matplotlib.pyplot as plt

# Calculate entropy per action for labeling
mean_entropy_per_action = reshaped_entropy.mean(dim=1)  # Average entropy across heads for each action

# Helper function to get the image indices around a specific action index (with boundaries check)
def get_surrounding_indices(idx, num_actions, window=6):
    return [i for i in range(max(0, idx - window), min(num_actions, idx + window + 1), 2)]

# Plot function to visualize images around a specific index
def plot_entropy_images(indices, title):
    num_actions = len(all_imgs[seeds[-1]])
    for main_idx in indices:
        surrounding_indices = get_surrounding_indices(main_idx, num_actions)
        
        plt.figure(figsize=(20, 5))
        for i, idx in enumerate(surrounding_indices):
            img = all_imgs[seeds[-1]][idx]
            entropy = mean_entropy_per_action[idx].item()
            
            plt.subplot(1, len(surrounding_indices), i + 1)
            plt.imshow(img)
            plt.title(f"Action {idx}\nEntropy: {entropy:.4f}")
            plt.axis('off')
        
        plt.suptitle(f"{title} - Center Action {main_idx} (Entropy: {mean_entropy_per_action[main_idx].item():.4f})", fontsize=16)
        plt.show()

# Visualize lowest entropy actions with surrounding images
plot_entropy_images(lowest_entropy_indices, "Lowest Entropy Actions")

# Visualize highest entropy action with surrounding images
plot_entropy_images([123, 24, 76], "Highest Entropy Action")


# %%

# Now you can use these indices to visualize the corresponding images from nimgs
lowest_entropy_images = [all_imgs[seeds[-1]][idx] for idx in lowest_entropy_indices]
highest_entropy_images = [all_imgs[seeds[-1]][idx] for idx in highest_entropy_indices]

# Example visualization
import matplotlib.pyplot as plt

# Plot lowest entropy images
plt.figure(figsize=(15, 5))
for i, img in enumerate(lowest_entropy_images):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Lowest Entropy Action {lowest_entropy_indices[i]}")
    plt.axis('off')
plt.show()

# Plot highest entropy images
plt.figure(figsize=(15, 5))
for i, img in enumerate(highest_entropy_images):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Highest Entropy Action {highest_entropy_indices[i]}")
    plt.axis('off')
plt.show()



# %% 
from IPython.display import Video
vwrite('lowdim1_score1_highent.mp4', all_imgs[seeds[-1]][118:128])
Video('lowdim1_score1_highent.mp4', embed=True, width=256, height=256)


# %%
from IPython.display import Video
vwrite('lowdim1_score0.mp4', all_imgs[seeds[0]])
Video('lowdim1_score0.mp4', embed=True, width=256, height=256)


# %%

# Saving dictionaries
torch.save(all_imgs, 'all_imgs.pt')
torch.save(all_obs, 'all_obs.pt')

# %%
# Loading dictionaries
loaded_dict1 = torch.load('all_imgs.pt')
loaded_dict2 = torch.load('all_obs.pt')

# Verify the loaded content
print("Loaded dict1:", loaded_dict1)
print("Loaded dict2:", loaded_dict2)

# %%
        
reshaped_entropy_multi =  torch.stack([action_entropy_self[:, head, :].reshape(-1) for head in range(4)], dim=0).T
reshaped_entropy_multi.shape

# %%

plot_entropy_over_actions(reshaped_entropy_multi, "Self", 4, reshaped_entropy_multi.shape[0])


# %%

# # Stack collected entropies and plot per head
# action_entropy_multi = torch.stack(action_entropy_multi, dim=0)  # Shape: [num_actions, num_heads, seq_length]
# action_entropy_self = torch.stack(action_entropy_self, dim=0)

plot_entropy_over_actions(action_entropy_multi.mean(dim=-1), "Multi-Head", 4, len(action))
plot_entropy_over_actions(action_entropy_self.mean(dim=-1), "Self", 4, len(action))


# %%
max_steps = 200
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100000)

# get first observation
obs = env.reset()
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
import matplotlib.pyplot as plt

img = imgs[0]
img_shape = (96, 96, 3)
x_coords = obs[0:20:2]
y_coords = obs[1:20:2]

plt.figure(figsize=(6, 6))
plt.imshow(img.astype(np.uint8))

# Plot each point and number it
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    plt.plot(x / 512 * img_shape[1], y / 512 * img_shape[0], 'ro')  # Normalize to fit image shape
    plt.text(x / 512 * img_shape[1], y / 512 * img_shape[0], str(i + 1), color="blue", fontsize=12)

plt.title("Key Points from Observation Plotted on Image")
plt.axis('off')
plt.show()

# %%
import torch
import torch.nn.functional as F
import math
max_steps = 200
env.seed(1000655)
obs = env.reset()
imgs = [env.render(mode='rgb_array')]

# attention_patterns = []
# def attention_hook(module, input, output):
#     query, key, value = input[0], input[1], input[2]
#     in_proj_weight = module.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
#     in_proj_bias = module.in_proj_bias      # Shape: (3*embed_dim)
#     embed_dim = module.embed_dim
#     num_heads = module.num_heads
#     head_dim = embed_dim // num_heads

#     # Split the projection weights and biases
#     q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
#     q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
#     q = F.linear(query, q_proj_weight, q_proj_bias)  # Shape: (batch_size, seq_length, embed_dim)
#     k = F.linear(key, k_proj_weight, k_proj_bias)
#     v = F.linear(value, v_proj_weight, v_proj_bias)

#     batch_size, seq_length, _ = q.size()

#     q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
#     k = k.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
#     v = v.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

#     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

#     attn_weights = F.softmax(scores, dim=-1)

#     attention_patterns.append(attn_weights.detach().cpu())

import torch
import torch.nn.functional as F
import math

attention_patterns = []

def self_attention_hook(module, input, output):
    # Unpack the inputs to handle cases with both self-attention and multihead attention
    query, key, value = input[0], input[1], input[2]
    
    # Get the projection weights and biases
    in_proj_weight = module.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
    in_proj_bias = module.in_proj_bias      # Shape: (3*embed_dim)
    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    # Split the projection weights and biases for query, key, and value
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
    
    # Apply linear transformations for query, key, and value
    q = F.linear(query, q_proj_weight, q_proj_bias)  # Shape: (batch_size, seq_length, embed_dim)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)
    print(q.shape, k.shape, v.shape)
    # Reshape and transpose to separate heads
    batch_size, seq_length, _ = q.size()
    _, enc_seq_length, _ = k.size()
    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)

    # Calculate scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Store the attention patterns for later inspection
    attention_patterns.append(attn_weights.detach().cpu())
import torch
import torch.nn.functional as F
import math

# A list to store attention patterns for later analysis
multi_attention_patterns = []
self_attention_patterns = []
def multi_attention_hook(module, input, output):
    # Unpack the inputs
    query, key, value = input[0], input[1], input[2]
    
    # Get the projection weights and biases
    in_proj_weight = module.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
    in_proj_bias = module.in_proj_bias      # Shape: (3*embed_dim)
    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    # Split the projection weights and biases for query, key, and value
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
    
    # Apply linear transformations for query, key, and value
    q = F.linear(query, q_proj_weight, q_proj_bias)  # Shape: (batch_size, seq_length, embed_dim)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)

    # Reshape and transpose to separate heads
    batch_size, seq_length, _ = q.size()
    _, enc_seq_length, _ = k.size()
    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)

    # Calculate scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Store the attention patterns for later inspection
    multi_attention_patterns.append(attn_weights.detach().cpu())

def self_attention_hook(module, input, output):
    # Unpack the inputs
    query, key, value = input[0], input[1], input[2]
    
    # Get the projection weights and biases
    in_proj_weight = module.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
    in_proj_bias = module.in_proj_bias      # Shape: (3*embed_dim)
    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    # Split the projection weights and biases for query, key, and value
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
    
    # Apply linear transformations for query, key, and value
    q = F.linear(query, q_proj_weight, q_proj_bias)  # Shape: (batch_size, seq_length, embed_dim)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)

    # Reshape and transpose to separate heads
    batch_size, seq_length, _ = q.size()
    _, enc_seq_length, _ = k.size()
    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, enc_seq_length, num_heads, head_dim).transpose(1, 2)

    # Calculate scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Store the attention patterns for later inspection
    self_attention_patterns.append(attn_weights.detach().cpu())

obs_deque = collections.deque(
    [obs[:20]] * obs_horizon, maxlen=obs_horizon)
obs_seq = np.stack(obs_deque)
nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)

multi_attn_module = policy.model.decoder.layers[7].multihead_attn
handle_multi = multi_attn_module.register_forward_hook(multi_attention_hook)

self_attn_module = policy.model.decoder.layers[7].self_attn
handle_self = self_attn_module.register_forward_hook(self_attention_hook)

try: 
    with torch.no_grad():
        nobs = policy.normalizer['obs'].normalize(nobs)
        B, _, Do = nobs.shape
        To = policy.n_obs_steps
        T = policy.horizon
        Da = policy.action_dim

        # build input
        device = policy.device
        dtype = policy.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        cond = nobs[:,:To]

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

        # set step values
        policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
        # print("init trakectory", trajectory)
        for t in policy.noise_scheduler.timesteps:
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = policy.noise_scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **policy.kwargs
                ).prev_sample
            
            # if t % 10==0: 
            #     print("timestep: ", t)
            #     print("trajectory: ", trajectory)
            #     print("model_output: ", model_output)
            #     print("cond: ", cond)
        
        # finally make sure conditioning is enforced
        trajectory[cond_mask] = cond_data[cond_mask] 
        print(trajectory.shape)
        # unnormalize prediction
        naction_pred = trajectory[...,:Da]
        action_pred = policy.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + policy.n_action_steps
        action = action_pred[:,start:end]
finally:
    handle_multi.remove()
    handle_self.remove()
    torch.cuda.empty_cache()
len(multi_attention_patterns)


# %%

multi_attention_patterns[0].shape


# %%


# Extract the number of heads
n_heads = multi_attention_patterns[0].shape[1]

# Create heatmaps for each head's attention pattern
for head in range(n_heads):
    plt.figure(figsize=(6, 5))
    plt.imshow(multi_attention_patterns[0][0, head].cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Attention Pattern for Head {head + 1}')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.show()

# %%

# Extract the number of heads
n_heads = self_attention_patterns[0].shape[1]

# Create heatmaps for each head's attention pattern
for head in range(n_heads):
    plt.figure(figsize=(6, 5))
    plt.imshow(self_attention_patterns[0][0, head].cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Attention Pattern for Head {head + 1}')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.show()


# %%
    
# List to store entropies at each time step
entropy_over_time = []

for t, attention_weights in enumerate(self_attention_patterns):
    # Ensure attention_weights are probabilities (if not already)
    # If attention_weights are outputs from softmax, this step is not necessary
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

    # Compute entropy per head and query position
    # Shape: [batch_size, num_heads, seq_length]
    entropies = -torch.sum(attention_weights * torch.log(attention_weights + 1e-6), dim=-1)
    
    # Optionally aggregate over heads
    # entropy_per_position = entropies.mean(dim=1)  # Shape: [batch_size, seq_length]
    
    # Store entropies for this time step
    entropy_over_time.append(entropies)  # Shape: [batch_size, num_heads, seq_length]

# Stack entropies over time
# Shape: [time_steps, batch_size, num_heads, seq_length]
entropy_over_time = torch.stack(entropy_over_time)

# %%

# Mean over batch and sequence positions
mean_entropy_over_time = entropy_over_time.mean(dim=(1, 3))  # Shape: [time_steps, num_heads]
import matplotlib.pyplot as plt

def plot_mean_entropy_over_time(mean_entropy_over_time):
    time_steps = mean_entropy_over_time.shape[0]
    num_heads = mean_entropy_over_time.shape[1]
    plt.figure(figsize=(12, 6))
    for head in range(num_heads):
        plt.plot(range(time_steps), mean_entropy_over_time[:, head], label=f'Head {head+1}')
    plt.title('Mean Attention Entropy Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_mean_entropy_over_time(mean_entropy_over_time)

# %%
# Modify the function to show token-specific entropy variations for each head
def plot_token_specific_entropy_over_time(entropy_over_time):
    time_steps, _, num_heads, seq_length = entropy_over_time.shape
    plt.figure(figsize=(12, 6 * num_heads))
    
    for head in range(num_heads):
        plt.subplot(num_heads, 1, head + 1)
        for token in range(seq_length):
            plt.plot(range(time_steps), entropy_over_time[:, 0, head, token], label=f'Token {token+1}')
        
        plt.title(f'Entropy Variation Over Time for Head {head + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        plt.legend(title="Tokens", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Plotting the token-specific entropy for each head over time
plot_token_specific_entropy_over_time(entropy_over_time)

# %%

import torch

def compute_token_wise_entropy_over_time(attention_patterns):
    """
    Computes token-wise entropies for each head, averaged over the time dimension.

    Parameters:
    - attention_patterns (list of torch.Tensor): List of attention pattern tensors, 
      where each tensor is of shape [batch_size, num_heads, seq_length, seq_length].

    Returns:
    - avg_token_entropy (torch.Tensor): A tensor of shape [num_heads, seq_length]
      representing token-wise entropies for each head averaged over time.
    """
    # List to store entropies at each time step
    entropy_over_time = []

    for attention_weights in attention_patterns:
        # Ensure attention_weights are probabilities (if not already)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        # Compute entropy per head and query position (over keys)
        entropies = -torch.sum(attention_weights * torch.log(attention_weights + 1e-6), dim=-1)
        
        # Append entropy for this time step
        entropy_over_time.append(entropies)  # Shape: [batch_size, num_heads, seq_length]

    # Stack entropies over time and average across the time dimension
    entropy_over_time = torch.stack(entropy_over_time)  # Shape: [time_steps, batch_size, num_heads, seq_length]
    avg_token_entropy = entropy_over_time.mean(dim=0).mean(dim=0)  # Shape: [num_heads, seq_length]

    return avg_token_entropy

avg_token_entropy = compute_token_wise_entropy_over_time(self_attention_patterns)

print("Average Token-Wise Entropy for Each Head:", avg_token_entropy)




# %%

try_action = env.action_space.sample()

observation, reward, done, info = env.step(try_action)

# %%
observation



# %%

# env = PushTEnv()
# limit enviornment interaction to 200 steps before termination
max_steps = 200
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100)

# get first observation
obs= env.reset()

obs.shape

# %%

obs_horizon = 10
# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

# %%
policy.to('cuda')
obs_seq = np.stack(obs_deque)
nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)
with torch.no_grad():
    action = policy.predict_action({
        'obs': nobs
    })
    naction = action['action']


# %%
seed = 1000432
env.seed(seed)
max_steps = 300
# get first observation
obs = env.reset()
obs_horizon = 10
# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs[:20]] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        obs_seq = np.stack(obs_deque)
        # infer action
        nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)
        print(nobs)
        with torch.no_grad():
            action = policy.predict_action({
                'obs': nobs
            })
            print(action)
            naction = action['action']
        naction = naction.detach().to('cpu').numpy()
        action = naction[0]
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _ = env.step(action[i])
            # save observations
            # print(obs)
            obs_deque.append(obs[:20])
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break
print('Score: ', max(rewards))

# visualize
from IPython.display import Video
vwrite('lowdim1.mp4', imgs)
Video('lowdim1.mp4', embed=True, width=256, height=256)
























# %%

#@markdown ### **Dataset Demo**

# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)




# %%
stats['obs']

# %%

policy.normalizer

# %%
obs_horizon = config['policy']['horizon']
obs_dim = 5
action_dim = 2

# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100000)

# get first observation
obs, info = env.reset()

obs
# %%
obs_horizon = 4
# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

# %%
obs_seq = np.stack(obs_deque)
nobs = policy.normalizer['obs'].normalize(obs_seq)
# nobs.shape

# nobs = normalize_data(obs_seq, stats=stats['obs'])
# nobs.shape

# add a batch dimension to obs_seq
nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)
nobs.shape

# %%

policy.normalizer['obs'].params_dict.input_stats.max

# %%

stats['obs']


# %%

obs_seq = np.stack(obs_deque)
nobs = torch.from_numpy(obs_seq).unsqueeze(0).to('cuda', dtype=torch.float32)
with torch.no_grad():
    action = policy.predict_action({
        'obs': nobs
    })
    naction = action['action']

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        # nobs = normalize_data(obs_seq, stats=stats['obs'])
        # device transfer
        # nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            action = policy.predict_action({
                'obs': obs_seq
            })
            naction = action['action']
            # # reshape observation to (B,obs_horizon*obs_dim)
            # obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # # initialize action from Guassian noise
            # noisy_action = torch.randn(
            #     (B, pred_horizon, action_dim), device=device)
            # naction = noisy_action

            # # init scheduler
            # noise_scheduler.set_timesteps(num_diffusion_iters)

            # for k in noise_scheduler.timesteps:
            #     # predict noise
            #     noise_pred = ema_noise_pred_net(
            #         sample=naction,
            #         timestep=k,
            #         global_cond=obs_cond
            #     )

            #     # inverse diffusion step (remove noise)
            #     naction = noise_scheduler.step(
            #         model_output=noise_pred,
            #         timestep=k,
            #         sample=naction
            #     ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print('Score: ', max(rewards))























# %%

model_state_dict = state_dict['state_dicts']['model'].keys()

# %%

config.keys()


# %% 
config['policy'].keys()


# %%
config['policy']['model'].keys()

# %%
for k, v in config.items():
    print(k, v)







# %%

from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
ckpt_path = "/work/pi_hzhang2_umass_edu/jnainani_umass_edu/Interp4Robotics/diffusionInterp/data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt"
ckpt_path = "/work/pi_hzhang2_umass_edu/jnainani_umass_edu/Interp4Robotics/diffusionInterp/data/experiments/image/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt"
state_dict = torch.load(ckpt_path, map_location='cuda')
# Access configuration, state dicts, and metadata
config = state_dict['cfg']
model_state_dict = state_dict['state_dicts']['model']
ema_state_dict = state_dict['state_dicts'].get('ema_model', None)
optimizer_state_dict = state_dict['state_dicts']['optimizer']
metadata = state_dict['pickles']

num_diffusion_iters = 100


noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_start= 0.0001,
    beta_end=0.02,
    beta_schedule='squaredcos_cap_v2',
    variance_type="fixed_small",
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# Initialize the model with configurations from `config`
model_t = DiffusionTransformerHybridImagePolicy(
    shape_meta=config['shape_meta'],
    noise_scheduler=noise_scheduler,
    horizon=config['horizon'],
    n_action_steps=config['n_action_steps'],
    n_obs_steps=config['n_obs_steps'],
    obs_as_cond= config['obs_as_cond'],
    crop_shape = (84, 84),
    obs_encoder_group_norm = True,
    # eval_fixed_crop = True,
    n_layer=8,
    n_head=4,
    n_emb=256,
    p_drop_attn=0.3,
    p_drop_emb=0.0,
    causal_attn=True,
    time_as_cond=True, 
    n_cond_layers=0,
    # Add other relevant parameters from `config`
)

# Load the state dict into the model with strict=False
model_t.load_state_dict(model_state_dict, strict=False)

model_t = model_t.to("cuda")  # Move the model to GPU

# %%

# %%
#@markdown ### **Dataset Demo**

# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['image'].shape:", batch['image'].shape)
print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
print("batch['action'].shape", batch['action'].shape)




# %%

with torch.no_grad():
    image = torch.zeros((1, obs_horizon,3,96,96))
    agent_pos = torch.zeros((1, obs_horizon, 2))
    # model.obs_encoder({"obs":image})
    obs_dict = {"image":image.cpu(), "agent_pos":agent_pos.cpu()}
    res = model_t.predict_action(obs_dict)