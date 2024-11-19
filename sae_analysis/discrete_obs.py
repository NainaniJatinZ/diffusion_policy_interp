# %%
import os
os.chdir("../")
from sae.sae_model import SparseAutoencoder
import torch 
import pandas as pd 
from tqdm.auto import tqdm
from skvideo.io import vwrite
from IPython.display import Video
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import collections
import random
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed1_acts = pd.read_csv("data/activations_summary.csv")
seed2_acts = pd.read_csv("data/activations_summary_2.csv")

sae_path = "sae/results_layer4_dim2048_k64_auxk64_dead200/checkpoints/last.ckpt"
sae_weights = torch.load(sae_path)
ckpt = {}
for k in sae_weights['state_dict'].keys():
    if k.startswith('sae_model.'):
        ckpt[k.split(".")[1]] = sae_weights['state_dict'][k]
sae = SparseAutoencoder(256, 2048, 64, 64, 32, 200)
sae.load_state_dict(ckpt)
sae.to(device)

ckpt_path = "/work/pi_hzhang2_umass_edu/jnainani_umass_edu/Interp4Robotics/diffusionInterp/data/experiments/low_dim/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
state_dict = torch.load(ckpt_path, map_location=device)
config = state_dict['cfg']
model_config = config['policy']['model']
model_config = {k: v for k, v in model_config.items() if not k.startswith('_target_')}
model = TransformerForDiffusion(**model_config)
noise_scheduler_config = config['policy']['noise_scheduler']
noise_scheduler = DDPMScheduler(**noise_scheduler_config)
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
policy = DiffusionTransformerLowdimPolicy(**policy_params)
policy.load_state_dict(state_dict['state_dicts']['model'])
policy.to(device)

# %%

feature_idx = 922 

top_10_activations = seed2_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(1)
seed, step, timestep, _ = top_10_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']].values[0]
# %%
target_seed = int(seed)
target_step = int(step)
obs_horizon = 2
target_timestep = int(timestep)
env = PushTKeypointsEnv()
env.seed(target_seed)
np.random.seed(target_seed)
torch.manual_seed(target_seed)
obs = env.reset()
imgs = []  #
obs_deque = collections.deque([obs[:20]] * obs_horizon, maxlen=obs_horizon)
done = False
max_steps = 200
step_idx = 0
rewards = []
policy.to('cuda')
# inference_inputs = {}
with tqdm(total=max_steps, desc=f"Seed {target_seed} Eval") as pbar:
    while not done:
        B = 1
        obs_seq = np.stack(obs_deque)
        nobs = torch.from_numpy(obs_seq).unsqueeze(0).to(device, dtype=torch.float32)

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
                # step_input = {
                #     "timestep": t,
                #     "trajectory_input": trajectory.clone().cpu().numpy().tolist(),
                #     "cond_input": cond.clone().cpu().numpy().tolist()
                # }
                
                # Append to this timestep's entry in the seed dictionary
                # inference_inputs[step_idx].append(step_input)

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

            # inference_inputs[step_idx] = []
            if step_idx == target_step:
                target_obs = cond.clone().cpu().numpy()
                target_step_image = env.render(mode='rgb_array')
                target_action = action.clone().cpu().numpy()
                print(f"Step {step_idx} Image")

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
                print(f"Reward for {target_seed}: ", max(rewards))
                break


# %%
            
target_step_image.shape
# %%

policy.normalizer['obs'].unnormalize(target_obs).shape

# %%

import torch
import matplotlib.pyplot as plt
import numpy as np

# Example inputs
image = target_step_image  # (96, 96, 3)
obs = policy.normalizer['obs'].unnormalize(target_obs)[0][0]  # torch.Size([1, 2, 20])

# Extract the points in (x, y) format
# Reshape the last dimension into 10 pairs of (x, y)
points = np.array(obs.detach().cpu()).reshape(-1, 2)

# Rescale points from [0, 512] to [0, 96]
points = points / 512 * 96
target_action_norm = target_action[0] / 512 * 96
# Center of the image
center_x, center_y = 96 / 2, 96 / 2

# Distance and angle
d_i = 30
angle = (5 * np.pi) / 4  # 45 degrees

# Endpoint of the line
end_x = center_x + d_i * np.cos(angle)
end_y = center_y - d_i * np.sin(angle)  # Subtract because y-coordinates increase downwards in images

# Plot the image
plt.imshow(image)
plt.axis('off')

# Overlay the points and annotate them
for idx, (x, y) in enumerate(points):
    plt.scatter(x, y, c='red', label=f'Point {idx}' if idx == 0 else None)  # Points now match image size
    plt.text(x, y, f'{idx}', fontsize=9, color='blue')

for idx, (x, y) in enumerate(target_action_norm):
    plt.scatter(x, y, c='yellow', label=f'Point {idx}' if idx == 0 else None)  # Points now match image size
    plt.text(x, y, f'{idx}', fontsize=9, color='red')

# Plot the center of the image
plt.scatter(center_x, center_y, c='green', label='Center')  # Center of the image

# plt.scatter(1, 1, c='yellow', label='Action')  # Center of the image

# Draw the line
plt.plot([center_x, end_x], [center_y, end_y], c='purple', label=f'Line d_i={d_i}, angle=pi/4')

plt.legend(loc='upper left')
plt.title("Points and Line Overlay on Image")
plt.show()

# %%

mid_pt = points[3]
low_t = points[4]
curr_block_theta = np.arctan2(- mid_pt[1] + low_t[1], mid_pt[0] - low_t[0])
print(curr_block_theta)
print(curr_block_theta * 180 / np.pi)

# %%

def theta_tc(obs):
    """
    Calculates the angle between the target block and the current block
    """
    mid_pt = obs[3]
    low_t = obs[4]
    curr_block_theta = np.arctan2(- mid_pt[1] + low_t[1], mid_pt[0] - low_t[0])
    theta_tc = curr_block_theta - np.pi / 4
    return theta_tc, theta_tc * 180 / np.pi, curr_block_theta, curr_block_theta * 180 / np.pi

theta_tc(points)

# %%

def dist_tc(obs):
    """
    Calculates the distance between the target block and the current block
    """
    mid_pt_current = obs[3]
    mid_pt_target = [48, 48]
    dist = np.sqrt((mid_pt_current[0] - mid_pt_target[0])**2 + (mid_pt_current[1] - mid_pt_target[1])**2)
    return dist

dist_tc(points)

# %%

def Ka(obs):
    """
    Calculates the closest keypoint to agent """
    agent = obs[-1]
    min_dist = np.inf
    min_idx = -1
    for idx, (x, y) in enumerate(obs):
        if idx == 9:
            continue
        else:
            dist = np.sqrt((x - agent[0])**2 + (y - agent[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
    return min_idx

Ka(points)


# %%

def theta_action(action):
    """
    Calculates the change in angle of the first and last action
    """
    first_angle = np.arctan2(- action[0][1] + action[1][1], action[0][0] - action[1][0])
    last_angle = np.arctan2(- action[-2][1] + action[-1][1], action[-2][0] - action[-1][0])
    change = last_angle - first_angle
    return first_angle, last_angle, change, change * 180 / np.pi

theta_action(target_action_norm)

# %%

def dist_action(action):
    """
    Calculates the distance between the first and last action
    """
    dist = np.sqrt((action[0][0] - action[-1][0])**2 + (action[0][1] - action[-1][1])**2)
    return dist

dist_action(target_action_norm)


# %%

def dist_action_mid(action, obs):
    """
    Calculates the average distance between actions and midpoint of target block
    """
    mid_pt_target = obs[3]
    dist = 0
    for idx, (x, y) in enumerate(action):
        dist += np.sqrt((x - mid_pt_target[0])**2 + (y - mid_pt_target[1])**2)
    return dist / len(action)

dist_action_mid(target_action_norm, points)
    
# %%
