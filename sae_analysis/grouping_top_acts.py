# %%
import os
os.chdir("../")
from sae.sae_model import SparseAutoencoder
import torch 
import pandas as pd 

seed1_acts = pd.read_csv("data/activations_summary.csv")
seed2_acts = pd.read_csv("data/activations_summary_2.csv")

# %%
feature_idx = 922
top_10_activations = seed1_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(10)

top_10_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

# %%

top_10_activations = seed2_acts.sort_values(
    f"feature_{feature_idx}", ascending=False
).head(10)

top_10_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']]

# %%

f36_0 = top_10_activations[['seed', 'step_idx', 'timestep', 'feature_36']].iloc[0].values

inf_inputs = torch.load(f"data/inference_inputs_seed_{int(f36_0[0].item())}.pt")

f36_first = inf_inputs[int(f36_0[1].item())][0]

# %%

f36_first['trajectory_input'][0]




# %%
sae_path = "sae/results_layer4_dim2048_k64_auxk64_dead200/checkpoints/last.ckpt"
sae_weights = torch.load(sae_path)
ckpt = {}
for k in sae_weights['state_dict'].keys():
    if k.startswith('sae_model.'):
        ckpt[k.split(".")[1]] = sae_weights['state_dict'][k]
sae = SparseAutoencoder(256, 2048, 64, 64, 32, 200)
sae.load_state_dict(ckpt)
sae.to('cuda')

# %%
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = "/work/pi_hzhang2_umass_edu/jnainani_umass_edu/Interp4Robotics/diffusionInterp/data/experiments/low_dim/pusht/diffusion_policy_transformer/train_2/checkpoints/latest.ckpt"
state_dict = torch.load(ckpt_path, map_location='cuda')
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
policy.to('cuda')
# %%

top_10_activations = seed2_acts.sort_values(
    f"feature_{652}", ascending=False
).head(10)

top_10_activations[['seed', 'step_idx', 'timestep', 'feature_36']]

f36_0 = top_10_activations[['seed', 'step_idx', 'timestep', 'feature_36']].iloc[0].values

inf_inputs = torch.load(f"data/inference_inputs_seed_{int(f36_0[0].item())}.pt")

f36_first = inf_inputs[int(f36_0[1].item())][0]

traj = torch.Tensor(f36_first['trajectory_input'][0])
cond = torch.Tensor(f36_first['cond_input'][0])

traj.shape, cond.shape

# %%
import matplotlib.pyplot as plt
# Convert each tensor of shape [20] into pairs of coordinates
coordinates_1 = policy.normalizer['obs'].unnormalize(cond)[0].reshape(-1, 2)  # First set of coordinates
coordinates_2 = policy.normalizer['obs'].unnormalize(cond)[1].reshape(-1, 2)  # Second set of coordinates

# Plot the points
plt.figure(figsize=(8, 8))
plt.scatter(coordinates_1[:, 0].detach().cpu(), coordinates_1[:, 1].detach().cpu(), color='blue', label='Set 1')
# plt.scatter(coordinates_2[:, 0], coordinates_2[:, 1], color='red', label='Set 2')

# Label each point with its index
for i, (x, y) in enumerate(coordinates_1):
    plt.text(x, y, str(i), color='blue', fontsize=10, ha='right', va='bottom')
    
# for i, (x, y) in enumerate(coordinates_2):
#     plt.text(x, y, str(i + 1), color='red', fontsize=10, ha='right', va='bottom')


# Customize plot
plt.xlim(0, 512)
plt.ylim(0, 512)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Plot of Coordinates')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
# %%
