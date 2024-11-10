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
policy.to('cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set seed for reproducibility
seed = 1000043
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

rewards = []
obs_list = [obs[:20]]  # To store observations for this seed
imgs = []  # To store images for this seed
# multi_attn_module = policy.model.decoder.layers[7].multihead_attn
# handle_multi = multi_attn_module.register_forward_hook(multi_attention_hook)

# self_attn_module = policy.model.decoder.layers[7].self_attn
# handle_self = self_attn_module.register_forward_hook(self_attention_hook)
# multi_attention_patterns = []
# self_attention_patterns = []

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
                # multi_entropy = compute_token_wise_entropy(multi_attention_patterns)
                # self_entropy = compute_token_wise_entropy(self_attention_patterns)
                # print(len(multi_attention_patterns))
                
                # unnormalize prediction
                naction_pred = trajectory[...,:Da]
                action_pred = policy.normalizer['action'].unnormalize(naction_pred)

                start = To - 1
                end = start + policy.n_action_steps
                action = action_pred[:,start:end]
                # Only keep relevant tokens for action steps
                # action_entropy_multi.append(multi_entropy[:, start:end])
                # action_entropy_self.append(self_entropy[:, start:end])
                # # Clear the lists for next sequence
                # multi_attention_patterns.clear()
                # self_attention_patterns.clear()
                # print(len(multi_attention_patterns))

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
    # handle_multi.remove()
    # handle_self.remove()
    torch.cuda.empty_cache()

# action_entropy_multi_seeds.append(torch.stack(action_entropy_multi, dim=0))
# action_entropy_self_seeds.append(torch.stack(action_entropy_self, dim=0))
# scores.append(max(rewards))
# Save observations and images for the current seed
# all_obs[seed] = obs_list
# all_imgs[seed] = imgs

print('Score: ', max(rewards))


# %%
from IPython.display import Video
vwrite('lowdim1.mp4', imgs)
Video('lowdim1.mp4', embed=True, width=256, height=256)
# %%
