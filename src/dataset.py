"""
Synthetic SAR Satellite Dataset Module for FloodLens AI.
Simulates dual-polarization Sentinel-1 SAR (VV/VH bands) radar backscatter
and binary flood inundation ground truth masks.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticSARFloodDataset(Dataset):
    """
    Simulates a paired SAR satellite dataset for semantic segmentation.
    Models physical radar backscatter characteristics:
      - Rough terrain/land: ~ -15 dB backscatter with diffuse scattering
      - Submerged flood water: ~ -25 dB due to specular reflection
      - Two polarization channels: VV (Vertical transmit/Vertical receive) and VH (Cross-polarization)

    Produces:
      - X: SAR image tensor of shape (2, img_size, img_size), dtype float32, normalized in [0, 1]
      - y: Binary flood mask tensor of shape (1, img_size, img_size), dtype float32 (1.0 = water, 0.0 = dry)
    """

    def __init__(
        self,
        num_samples: int = 128,
        img_size: int = 256,
        seed: Optional[int] = None,
        base_backscatter_mean: float = -15.0,
        base_backscatter_std: float = 3.0,
        water_attenuation: float = 10.0,
    ):
        """
        Args:
            num_samples: Total number of synthetic scenes to generate.
            img_size: Height and width in pixels (default: 256x256).
            seed: Optional seed for deterministic reproducibility.
            base_backscatter_mean: Mean land backscatter in dB.
            base_backscatter_std: Standard deviation of rough terrain backscatter.
            water_attenuation: Decibel reduction caused by specular reflection over calm flood water.
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed
        self.base_backscatter_mean = base_backscatter_mean
        self.base_backscatter_std = base_backscatter_std
        self.water_attenuation = water_attenuation

        # Seed random state for deterministic generation if provided
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single paired (SAR, Flood Mask) sample.
        """
        # If seed was provided, derive reproducible sample seed
        sample_rng = np.random.default_rng(self.seed + idx) if self.seed is not None else np.random.default_rng()

        # 1. Base background terrain: rough land backscatter (-15 dB mean, 3 dB std) across 2 channels
        sar = sample_rng.normal(
            self.base_backscatter_mean,
            self.base_backscatter_std,
            (2, self.img_size, self.img_size),
        ).astype(np.float32)

        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 2. Randomly generate a simulated flooded corridor (river overflow)
        min_start = max(10, int(self.img_size * 0.15))
        max_start = min(self.img_size - 40, int(self.img_size * 0.70))
        start_row = sample_rng.integers(min_start, max_start)
        band_width = sample_rng.integers(20, max(25, int(self.img_size * 0.25)))
        end_row = min(self.img_size, start_row + band_width)

        # Apply flood label to ground truth mask
        mask[start_row:end_row, :] = 1.0

        # 3. Simulate water specular reflection (drop backscatter signal by attenuation amount)
        sar[:, start_row:end_row, :] -= self.water_attenuation

        # 4. SAR normalization: scale physically from [-30 dB, 0 dB] range to [0, 1] range
        sar_normalized = np.clip((sar + 30.0) / 30.0, 0.0, 1.0)

        # Convert to PyTorch float32 tensors
        sar_tensor = torch.from_numpy(sar_normalized).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # Shape: (1, H, W)

        return sar_tensor, mask_tensor


def get_dataloaders(
    num_train: int = 128,
    num_val: int = 32,
    batch_size: int = 8,
    img_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience factory to create training and validation DataLoaders.
    """
    train_dataset = SyntheticSARFloodDataset(
        num_samples=num_train, img_size=img_size, seed=seed
    )
    val_dataset = SyntheticSARFloodDataset(
        num_samples=num_val, img_size=img_size, seed=seed + 1000
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader
