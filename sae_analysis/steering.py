# %% model and sae loading
import os
os.chdir("/work/pi_hzhang2_umass_edu/jnainani_umass_edu/diffusion_policy_interp/")
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

fstats = pd.read_csv(f"sae_analysis/out/f{feature_idx}_stats.csv")
print(fstats.head())
top_act_stats = list(fstats.iloc[0])
seed, step, timestep = top_act_stats[1], top_act_stats[2], top_act_stats[3]
print(seed, step, timestep)


# top_10_activations = seed2_acts.sort_values(
#     f"feature_{feature_idx}", ascending=False
# ).head(1)
# seed, step, timestep, _ = top_10_activations[['seed', 'step_idx', 'timestep', f'feature_{feature_idx}']].values[0]
# %%
target_seed = 92400 # int(seed)
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
                # print(f"Step {step_idx} Image")

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

from IPython.display import Video
vwrite(f'out/lowdim_seed{target_seed}_og.mp4', imgs)
Video(f'out/lowdim_seed{target_seed}_og.mp4', embed=True, width=256, height=256)

# %%

# target_seed = 37500 #int(seed)
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

def steer_latent(layer_num, sae, steering_coefficient, latent_idx):
    def hook(module, input, output):
        output = output + steering_coefficient * sae.w_dec[latent_idx]
        return output
    return hook
sfeature_idx = 1198
layer_num = 4
handle = policy.model.decoder.layers[layer_num].register_forward_hook(steer_latent(layer_num, sae, 6, sfeature_idx))
try: 
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
                    # print(f"Step {step_idx} Image")

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
finally:
    # Remove hooks
    handle.remove()
    torch.cuda.empty_cache()
    
from IPython.display import Video
vwrite(f'out/lowdim_seed{target_seed}_f{sfeature_idx}_steer.mp4', imgs)
Video(f'out/lowdim_seed{target_seed}_f{sfeature_idx}_steer.mp4', embed=True, width=256, height=256)

# %%