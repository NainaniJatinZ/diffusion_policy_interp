# %% model and sae loading
import os
import sys 
sys.path.append("sae")
sys.path.append("diffusion_policy")
sys.path.append("data")
# os.chdir("/work/pi_hzhang2_umass_edu/jnainani_umass_edu/diffusion_policy_interp")
print(os.getcwd())
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
    points_to_average = [
    obs[8], obs[1] #, obs[2]
    ]

    # Convert to a NumPy array for easy computation
    points_array = np.array(points_to_average)

    # Compute the average
    mid_pt = np.mean(points_array, axis=0)
    # mid_pt = obs[3]
    low_t = obs[4]
    curr_block_theta = np.arctan2(-(mid_pt[1] - low_t[1]), mid_pt[0] - low_t[0])
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

better_acts = pd.read_csv("data/better_activations_summary.csv")

import gym
from gym import spaces
from PIL import Image
import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import os
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions
import matplotlib.cm as cm

def get_image_from_obs(points, action, seed, step, timestep, action_mode="Input"):
    center_x, center_y = 96 / 2, 96 / 2
    d_i = 30
    angle = (5 * np.pi) / 4
    end_x = center_x + d_i * np.cos(angle)
    end_y = center_y - d_i * np.sin(angle)
    points_to_average = [
    points[8], points[1] 
    ]
    points_array = np.array(points_to_average)
    average_point = np.mean(points_array, axis=0)

    env = PushTKeypointsEnv()
    canvas = pygame.Surface((env.window_size, env.window_size))
    canvas.fill((255, 255, 255))
    env.screen = canvas
    draw_options = DrawOptions(canvas)
    # Draw goal pose.
    env.space = pymunk.Space()
    goal_body = env._get_goal_pose_body(np.array([256,256,np.pi/4]))
    env.goal_color = pygame.Color('LightGreen')
    block_temp = env.add_tee((256, 300), 0)
    for shape in block_temp.shapes:
        goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
        goal_points += [goal_points[0]]
        pygame.draw.polygon(canvas, env.goal_color, goal_points)
    img = np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )
    img = cv2.resize(img, (env.render_size, env.render_size))
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.axis('off')

    # Overlay the points and annotate them
    for idx, (x, y) in enumerate(points):
        if idx == 9:
            plt.scatter(x, y, c='blue', label='Agent start')
        else:
            plt.scatter(x, y, c='red', label=f'Point {idx}' if idx == 0 else None)  # Points now match image size
            plt.text(x, y, f'{idx}', fontsize=9, color='blue')

    # Plot arrows for the action trajectory with gradient colors
    num_actions = len(action) - 1
    colors = cm.viridis(np.linspace(0, 1, num_actions)) 
    for i in range(num_actions):
        start_x, start_y = action[i]
        end_x, end_y = action[i + 1]
        dx = end_x - start_x
        dy = end_y - start_y
        if i == 0:
            plt.quiver(start_x, start_y, dx, dy, angles='xy', scale_units='xy', scale=1, color=colors[i], label=action_mode+" start")
        elif i == num_actions - 1:
            plt.quiver(start_x, start_y, dx, dy, angles='xy', scale_units='xy', scale=1, color=colors[i], label=action_mode+" end")
        else:
            plt.quiver(start_x, start_y, dx, dy, angles='xy', scale_units='xy', scale=1, color=colors[i])

    top_left = points[1]     # Left side of the base
    top_right = points[2]    # Right side of the base
    center_top = points[4]   # Top of the T (center vertical bar)
    mid_point = average_point    # Middle connecting point
    # Draw the base of the T (horizontal line)
    plt.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], c='grey', linewidth=2)
    # Draw the vertical bar of the T
    plt.plot([mid_point[0], center_top[0]], [mid_point[1], center_top[1]], c='grey', linewidth=2)
    # Plot the center of the image
    plt.scatter(center_x, center_y, c='green', label='Center')  # 
    plt.legend(loc='upper left')
    # plt.title("Points and Line Overlay on Image")
    plt.savefig(f"sae_analysis/out/env_imgs/seed{seed}_step{step}_timestep{timestep}_action{action_mode}.png")
    plt.close()

# Define a function to compute statistics
def compute_stats(row, policy, save_img=True):
    seed, step, timestep, activation = row.values
    seed, step, timestep = int(seed), int(step), int(timestep)
    inf_inputs = torch.load(f"data/inference_inputs_seed_{int(seed)}.pt", weights_only=True)
    time_ind = 99 - int(timestep)
    obs = torch.Tensor(inf_inputs[int(step)][time_ind]['cond_input'])
    action_in = torch.Tensor(inf_inputs[int(step)][time_ind]['trajectory_input'])
    action_out = torch.Tensor(inf_inputs[int(step)][time_ind]['trajectory_output'])
    obs_norm1 = policy.normalizer['obs'].unnormalize(obs)[0][0]
    obs_norm2 = policy.normalizer['obs'].unnormalize(obs)[0][1]
    inp_action = np.array(policy.normalizer['action'].unnormalize(action_in)[0].detach().cpu()) / 512 * 96
    out_action = np.array(policy.normalizer['action'].unnormalize(action_out)[0].detach().cpu()) / 512 * 96
    points1 = np.array(obs_norm1.detach().cpu()).reshape(-1, 2) / 512 * 96
    points2 = np.array(obs_norm2.detach().cpu()).reshape(-1, 2) / 512 * 96
    _, theta_tc_deg, _, theta_c_deg = theta_tc(points2)
    d_tc = dist_tc(points2)
    ClosestK = Ka(points2)
    _, _, _, Delta_theta_act_in = theta_action(inp_action)
    _, _, _, Delta_theta_act_out = theta_action(out_action)
    Delta_dist_act_in = dist_action(inp_action)
    Delta_dist_act_out = dist_action(out_action)
    # print(dist_action_mid(inp_action, points1))
    d_act_curr_in = dist_action_mid(inp_action, points2)
    d_act_curr_out = dist_action_mid(out_action, points2)
    d_act_target_in = dist_action_target(inp_action)
    d_act_target_out = dist_action_target(out_action)

    img_saved = os.path.exists(f"sae_analysis/out/env_imgs/seed{seed}_step{step}_timestep{timestep}_actionInput.png")
    if save_img and not img_saved:
        get_image_from_obs(points2, inp_action, seed, step, timestep, "Input")
        get_image_from_obs(points2, out_action, seed, step, timestep, "Output")

    # Collect the results in a dictionary
    return {
        "seed": int(seed),
        "step_idx": int(step),
        "timestep": time_ind,
        "activation": activation,
        "theta_tc_deg": theta_tc_deg,
        "theta_c_deg": theta_c_deg,
        "d_tc": d_tc,
        "ClosestK": ClosestK,
        "Delta_theta_act_in": Delta_theta_act_in,
        "Delta_theta_act_out": Delta_theta_act_out,
        "Delta_dist_act_in": Delta_dist_act_in,
        "Delta_dist_act_out": Delta_dist_act_out,
        "d_act_curr_in": d_act_curr_in,
        "d_act_curr_out": d_act_curr_out,
        "d_act_target_in": d_act_target_in,
        "d_act_target_out": d_act_target_out,
        "img_path_in": f"sae_analysis/out/env_imgs/seed{seed}_step{step}_timestep{timestep}_actionInput.png",
        "img_path_out": f"sae_analysis/out/env_imgs/seed{seed}_step{step}_timestep{timestep}_actionOutput.png"
    }

def round_to_significant_digits(val):
    if isinstance(val, (int, float)):  # Check if the value is numeric
        return float(f"{val:.3g}")  # Format to 3 significant digits
    return val  

# Process each selected feature index
for feature_idx in range(1024, 2048):
    print(f"Processing feature index: {feature_idx}")
    results = []
    try:
        top_activations = better_acts.sort_values(
    f"feature_{feature_idx}", ascending=False).head(20)
    except Exception as e:
        print("Skipping feature ", feature_idx)
        continue
    latent_df = top_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]
    non_na_count = better_acts[f"feature_{feature_idx}"].notna().sum()
    if non_na_count < 50:
        print(f"Skipping feature {feature_idx} due to insufficient activations")
        continue
    for idx, row in latent_df.iterrows():
        stats = compute_stats(row, policy)
        results.append(stats)
    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results).sort_values("activation", ascending=False)
    # Reduce precision of numerical values to 3 significant digits
    results_df = results_df.apply(
        lambda col: col.map(round_to_significant_digits) if col.dtype in [np.float64, np.int64] else col
    )
    # Save the DataFrame to a CSV file
    results_df.to_csv(f"sae_analysis/out/new/f{feature_idx}_stats.csv", index=False)
    print(f"Saved CSV for feature index: {feature_idx}")
    # break