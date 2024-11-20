# %% model and sae loading
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


# %% helper functions


def theta_tc(obs):
    """
    Calculates the angle between the target block and the current block
    """
    mid_pt = obs[3]
    low_t = obs[4]
    curr_block_theta = np.arctan2(- mid_pt[1] + low_t[1], mid_pt[0] - low_t[0])
    theta_tc = curr_block_theta - np.pi / 4
    return theta_tc, theta_tc * 180 / np.pi, curr_block_theta, curr_block_theta * 180 / np.pi

def dist_tc(obs):
    """
    Calculates the distance between the target block and the current block
    """
    mid_pt_current = obs[3]
    mid_pt_target = [48, 48]
    dist = np.sqrt((mid_pt_current[0] - mid_pt_target[0])**2 + (mid_pt_current[1] - mid_pt_target[1])**2)
    return dist

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

def theta_action(action):
    """
    Calculates the change in angle of the first and last action
    """
    first_angle = np.arctan2(- action[0][1] + action[1][1], action[0][0] - action[1][0])
    last_angle = np.arctan2(- action[-2][1] + action[-1][1], action[-2][0] - action[-1][0])
    change = last_angle - first_angle
    return first_angle, last_angle, change, change * 180 / np.pi

def dist_action(action):
    """
    Calculates the distance between the first and last action
    """
    dist = np.sqrt((action[0][0] - action[-1][0])**2 + (action[0][1] - action[-1][1])**2)
    return dist

def dist_action_mid(action, obs):
    """
    Calculates the average distance between actions and midpoint of target block
    """
    mid_pt_target = obs[3]
    dist = 0
    for idx, (x, y) in enumerate(action):
        dist += np.sqrt((x - mid_pt_target[0])**2 + (y - mid_pt_target[1])**2)
    return dist / len(action)

def dist_action_target(action):
    """
    Calculates the average distance between actions and midpoint of target block
    """
    dist = 0
    for idx, (x, y) in enumerate(action):
        dist += np.sqrt((x - 48)**2 + (y - 48)**2)
    return dist / len(action)

# %% load data

feature_idx = 922 

top_10_activations = seed2_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(1)
seed, step, timestep, _ = top_10_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']].values[0]

# %%
inf_inputs = torch.load(f"data/inference_inputs_seed_{int(seed)}.pt")
# %%
obs = torch.Tensor(inf_inputs[int(step)][0]['cond_input'])
obs_norm1 = policy.normalizer['obs'].unnormalize(obs)[0][0]
obs_norm2 = policy.normalizer['obs'].unnormalize(obs)[0][1]
# %%
points1 = np.array(obs_norm1.detach().cpu()).reshape(-1, 2)  / 512 * 96
points2 = np.array(obs_norm2.detach().cpu()).reshape(-1, 2)  / 512 * 96
print(points1)
# %%
print(theta_tc(points1))
# %%
print(dist_tc(points1))
# %%
action = torch.Tensor(inf_inputs[int(step)][time_ind]['trajectory_input'])
np.array(policy.normalizer['action'].unnormalize(action)[0].detach().cpu()) / 512 * 96
# %%

feature_idx = 922 

top_10_activations = seed2_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(20)
latent_df = top_10_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

# iterate over latent df
for idx, row in latent_df.iterrows():
    seed, step, timestep, _ = row.values
    inf_inputs = torch.load(f"data/inference_inputs_seed_{int(seed)}.pt")
    time_ind = 99 - int(timestep)
    obs = torch.Tensor(inf_inputs[int(step)][time_ind]['cond_input'])
    action = torch.Tensor(inf_inputs[int(step)][time_ind]['trajectory_input'])
    obs_norm1 = policy.normalizer['obs'].unnormalize(obs)[0][0]
    obs_norm2 = policy.normalizer['obs'].unnormalize(obs)[0][1]
    inp_action = np.array(policy.normalizer['action'].unnormalize(action)[0].detach().cpu()) / 512 * 96
    points1 = np.array(obs_norm1.detach().cpu()).reshape(-1, 2)  / 512 * 96
    points2 = np.array(obs_norm2.detach().cpu()).reshape(-1, 2)  / 512 * 96
    print(f"Seed: {seed}, Step: {step}, Timestep: {timestep}")
    # print(theta_tc(points1))
    _, theta_tc_deg, _, theta_c_deg = theta_tc(points2)
    # print(dist_tc(points1))
    d_tc = dist_tc(points2)
    ClosestK = Ka(points2)
    # print(Ka(points2))
    _, _, _, Delta_theta_act = theta_action(inp_action)
    Delta_dist_act = dist_action(inp_action)
    # print(dist_action_mid(inp_action, points1))
    d_act_curr = dist_action_mid(inp_action, points2)
    d_act_target = dist_action_target(inp_action)
    print(f"Theta_tc: {theta_tc_deg}, d_tc: {d_tc}, ClosestK: {ClosestK}, Delta_theta_act: {Delta_theta_act}, Delta_dist_act: {Delta_dist_act}, d_act_curr: {d_act_curr}, d_act_target: {d_act_target}")
    print("\n")

# %%

import pandas as pd
import numpy as np
import torch

feature_idx = 922

# Define a function to compute statistics
def compute_stats(row, data_prefix, policy):
    seed, step, timestep, activation = row.values
    inf_inputs = torch.load(f"data/inference_inputs_seed_{int(seed)}.pt")
    time_ind = 99 - int(timestep)
    obs = torch.Tensor(inf_inputs[int(step)][time_ind]['cond_input'])
    action = torch.Tensor(inf_inputs[int(step)][time_ind]['trajectory_input'])
    obs_norm1 = policy.normalizer['obs'].unnormalize(obs)[0][0]
    obs_norm2 = policy.normalizer['obs'].unnormalize(obs)[0][1]
    inp_action = np.array(policy.normalizer['action'].unnormalize(action)[0].detach().cpu()) / 512 * 96
    points1 = np.array(obs_norm1.detach().cpu()).reshape(-1, 2) / 512 * 96
    points2 = np.array(obs_norm2.detach().cpu()).reshape(-1, 2) / 512 * 96
    _, theta_tc_deg, _, theta_c_deg = theta_tc(points2)
    d_tc = dist_tc(points2)
    ClosestK = Ka(points2)
    _, _, _, Delta_theta_act = theta_action(inp_action)
    Delta_dist_act = dist_action(inp_action)
    d_act_curr = dist_action_mid(inp_action, points2)
    d_act_target = dist_action_target(inp_action)

    # Collect the results in a dictionary
    return {
        "data_prefix": data_prefix,
        "seed": int(seed),
        "step_idx": int(step),
        "timestep": int(timestep),
        "activation": activation,
        "theta_tc_deg": theta_tc_deg,
        "theta_c_deg": theta_c_deg,
        "d_tc": d_tc,
        "ClosestK": ClosestK,
        "Delta_theta_act": Delta_theta_act,
        "Delta_dist_act": Delta_dist_act,
        "d_act_curr": d_act_curr,
        "d_act_target": d_act_target,
    }

# Create an empty list to store results
results = []

# Process seed2_acts
top_10_activations_seed2 = seed2_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(20)
latent_df_seed2 = top_10_activations_seed2[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

for idx, row in latent_df_seed2.iterrows():
    stats = compute_stats(row, "seed2", policy)
    results.append(stats)

# Process seed1_acts
top_10_activations_seed1 = seed1_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(20)
latent_df_seed1 = top_10_activations_seed1[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

for idx, row in latent_df_seed1.iterrows():
    stats = compute_stats(row, "seed1", policy)
    results.append(stats)

# Convert the results into a DataFrame
results_df = pd.DataFrame(results).sort_values("activation", ascending=False)
# Reduce precision of numerical values to 3 significant digits
def round_to_significant_digits(val):
    if isinstance(val, (int, float)):  # Check if the value is numeric
        return float(f"{val:.3g}")  # Format to 3 significant digits
    return val  # Return non-numeric values unchanged

# Apply rounding to all columns in the DataFrame
results_df = results_df.applymap(round_to_significant_digits)

results_df.to_csv(f"sae_analysis/out/f{feature_idx}_stats.csv", index=False)
# Display the DataFrame
# import ace_tools as tools; tools.display_dataframe_to_user(name="Feature Statistics DataFrame", dataframe=results_df)


# %%

import pandas as pd
import numpy as np
import torch
import random

# Define the feature list
feature_list = [i for i in range(0, 2047, 2)]

# Filter the feature list to include only indices with the column in both DataFrames
valid_features = [
    idx for idx in feature_list
    if f"feature_{idx}" in seed2_acts.columns and f"feature_{idx}" in seed1_acts.columns
]

# Randomly select a few indices from the valid feature list
random_indices = random.sample(valid_features, 10)  # Adjust the number as needed

print(f"Random indices with valid columns: {random_indices}")
# %%
# Define a function to compute statistics
def compute_stats(row, data_prefix, policy):
    seed, step, timestep, activation = row.values
    inf_inputs = torch.load(f"data/inference_inputs_seed_{int(seed)}.pt", weights_only=True)
    time_ind = 99 - int(timestep)
    obs = torch.Tensor(inf_inputs[int(step)][time_ind]['cond_input'])
    action = torch.Tensor(inf_inputs[int(step)][time_ind]['trajectory_input'])
    obs_norm1 = policy.normalizer['obs'].unnormalize(obs)[0][0]
    obs_norm2 = policy.normalizer['obs'].unnormalize(obs)[0][1]
    inp_action = np.array(policy.normalizer['action'].unnormalize(action)[0].detach().cpu()) / 512 * 96
    points1 = np.array(obs_norm1.detach().cpu()).reshape(-1, 2) / 512 * 96
    points2 = np.array(obs_norm2.detach().cpu()).reshape(-1, 2) / 512 * 96
    _, theta_tc_deg, _, theta_c_deg = theta_tc(points2)
    d_tc = dist_tc(points2)
    ClosestK = Ka(points2)
    _, _, _, Delta_theta_act = theta_action(inp_action)
    Delta_dist_act = dist_action(inp_action)
    d_act_curr = dist_action_mid(inp_action, points2)
    d_act_target = dist_action_target(inp_action)

    # Collect the results in a dictionary
    return {
        "data_prefix": data_prefix,
        "seed": int(seed),
        "step_idx": int(step),
        "timestep": int(timestep),
        "activation": activation,
        "theta_tc_deg": theta_tc_deg,
        "theta_c_deg": theta_c_deg,
        "d_tc": d_tc,
        "ClosestK": ClosestK,
        "Delta_theta_act": Delta_theta_act,
        "Delta_dist_act": Delta_dist_act,
        "d_act_curr": d_act_curr,
        "d_act_target": d_act_target,
    }

# Process each selected feature index
for feature_idx in random_indices:
    print(f"Processing feature index: {feature_idx}")
    
    results = []

    # Process seed2_acts
    top_10_activations_seed2 = seed2_acts.sort_values(
        f"feature_{feature_idx}", ascending=False
    ).head(20)
    latent_df_seed2 = top_10_activations_seed2[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

    for idx, row in latent_df_seed2.iterrows():
        stats = compute_stats(row, "seed2", policy)
        results.append(stats)

    # Process seed1_acts
    top_10_activations_seed1 = seed1_acts.sort_values(
        f"feature_{feature_idx}", ascending=False
    ).head(20)
    latent_df_seed1 = top_10_activations_seed1[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

    for idx, row in latent_df_seed1.iterrows():
        stats = compute_stats(row, "seed1", policy)
        results.append(stats)

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results).sort_values("activation", ascending=False)

    # Reduce precision of numerical values to 3 significant digits
    results_df = results_df.apply(
        lambda col: col.map(round_to_significant_digits) if col.dtype in [np.float64, np.int64] else col
    )

    # Save the DataFrame to a CSV file
    results_df.to_csv(f"sae_analysis/out/f{feature_idx}_stats.csv", index=False)
    print(f"Saved CSV for feature index: {feature_idx}")



# %%