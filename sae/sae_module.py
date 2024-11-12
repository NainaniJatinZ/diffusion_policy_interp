import pytorch_lightning as pl
import torch
from sae_model import SparseAutoencoder, loss_fn

class SAELightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.sae_model = SparseAutoencoder(
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
        )

    def forward(self, x):
        return self.sae_model(x)

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(0)
        recons, auxk, num_dead = self(batch)
        mse_loss, auxk_loss = loss_fn(batch, recons, auxk)
        loss = mse_loss + auxk_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("train_auxk_loss", auxk_loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("num_dead_neurons", num_dead, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.size(0)
        with torch.no_grad():
            recons, auxk, num_dead = self(batch)
            mse_loss, auxk_loss = loss_fn(batch, recons, auxk)
            loss = mse_loss + auxk_loss

        # Log only MSE loss and overall loss for validation
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_mse_loss", mse_loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def on_after_backward(self):
        self.sae_model.norm_weights()
        self.sae_model.norm_grad()
