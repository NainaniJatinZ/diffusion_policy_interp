import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

# Load all layers' activations from the saved file
# all_layers_activations = torch.load('../data/all_layers_activations_10seeds.pt')

class ActivationDataset(Dataset):
    def __init__(self, activations):
        self.activations = activations

    def __len__(self):
        return self.activations.size(0)

    def __getitem__(self, idx):
        return self.activations[idx]  # Returns a single activation tensor of shape [256]

class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, all_layers_activations, layer, batch_size):
        super().__init__()
        self.layer = layer
        self.batch_size = batch_size

        # Extract activations for the specified layer
        self.activations = all_layers_activations[f"layer_{layer}"]

    def setup(self, stage=None):
        # Calculate sizes for train, validation, and test splits
        total_size = len(self.activations)
        val_test_split = int(0.1 * total_size)  # 10% for validation, 10% for test
        train_size = total_size - 2 * val_test_split  # 80% for training

        # Split the dataset
        full_dataset = ActivationDataset(self.activations)
        self.train_data, self.val_data, self.test_data = random_split(
            full_dataset, [train_size, val_test_split, val_test_split]
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
