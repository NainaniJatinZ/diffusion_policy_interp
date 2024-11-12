# %%
import argparse
import os
os.chdir('../sae/')
import wandb
import torch
import pytorch_lightning as pl
from data_module import SequenceDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sae_module import SAELightningModule
# Set up argument parsing
parser = argparse.ArgumentParser(description="Sparse Autoencoder Training Script")

# Define arguments
parser.add_argument("--data-dir", type=str, default="../data/all_layers_activations_10seeds.pt", help="Path to activation data file")
parser.add_argument("--layer", type=int, default=4, help="Layer to use for activations")
parser.add_argument("--d-model", type=int, default=256, help="Input dimension of activations")
parser.add_argument("--d-hidden", type=int, required=True, help="Hidden dimension in autoencoder")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--k", type=int, required=True, help="Sparsity constraint")
parser.add_argument("--auxk", type=int, default=256, help="Auxiliary sparsity constraint")
parser.add_argument("--dead-steps-threshold", type=int, default=2000, help="Threshold for dead neurons")
parser.add_argument("--max-epochs", type=int, default=4, help="Maximum number of training epochs")
parser.add_argument("--num-devices", type=int, default=1, help="Number of GPUs to use")

# Parse arguments
args = parser.parse_args()
# parser = argparse.ArgumentParser()

# parser.add_argument("--data-dir", type=str, default="data/activations")
# parser.add_argument("-l", "--layer", type=int, default=4, help="Layer to use for activations")
# parser.add_argument("--d-model", type=int, default=256, help="Input dimension of activations")
# parser.add_argument("--d-hidden", type=int, default=32768, help="Hidden dimension in autoencoder")
# parser.add_argument("-b", "--batch-size", type=int, default=4)
# parser.add_argument("--lr", type=float, default=2e-4)
# parser.add_argument("--k", type=int, default=128)
# parser.add_argument("--auxk", type=int, default=256)
# parser.add_argument("--dead-steps-threshold", type=int, default=2000)
# parser.add_argument("-e", "--max-epochs", type=int, default=4)
# parser.add_argument("-d", "--num-devices", type=int, default=1)

# %%
# args = parser.parse_args()
# Define args as a dictionary
# Convert args_dict to an argparse.Namespace object for dot notation
# args = argparse.Namespace(**args_dict)

args.output_dir = f"results_layer{args.layer}_dim{args.d_hidden}_k{args.k}_auxk{args.auxk}_dead{args.dead_steps_threshold}"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# Initialize WandB logger
sae_name = f"layer{args.layer}_sae{args.d_hidden}_k{args.k}_auxk{args.auxk}_dead{args.dead_steps_threshold}"
wandb_logger = WandbLogger(
    project="diffusion_interp",
    name=sae_name,
    save_dir=os.path.join(args.output_dir, "wandb"),
)

# %%

# Initialize model and data module
model = SAELightningModule(args)
all_layers_activations = torch.load('../data/all_layers_activations_20seeds.pt')
# Instantiate the data module for the specified layer
data_module = SequenceDataModule(all_layers_activations, layer=args.layer, batch_size=args.batch_size)

# Set up checkpoints
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.output_dir, "checkpoints"),
    filename=sae_name + "-{step}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min",
    save_last=True,
)

# %%

# Trainer setup
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu",
    devices=list(range(args.num_devices)),
    strategy="ddp" if args.num_devices > 1 else "auto",
    logger=wandb_logger,
    log_every_n_steps=10,
    val_check_interval=2000,
    limit_val_batches=10,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

# Run training and testing
trainer.fit(model, data_module)
trainer.test(model, data_module)

wandb.finish()
# %%
