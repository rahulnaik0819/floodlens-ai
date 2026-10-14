"""
Neural Segmentation Model Module for FloodLens AI.
Implements U-Net architecture with ResNet-34 backbone for dual-polarization SAR imagery
and Dice loss computation.
"""

import os
from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp


def get_default_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = None,
    in_channels: int = 2,
    classes: int = 1,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Instantiate a U-Net semantic segmentation network.

    Args:
        encoder_name: Backbone encoder architecture (default: 'resnet34').
        encoder_weights: Pretrained weights name or None for training from scratch.
        in_channels: 2 channels corresponding to Sentinel-1 VV and VH polarizations.
        classes: 1 output channel for binary flood mask logits.
        device: Target execution device (CPU or CUDA).
    """
    if device is None:
        device = get_default_device()

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model.to(device)


def get_loss_fn(
    mode: str = smp.losses.BINARY_MODE,
    from_logits: bool = True,
) -> nn.Module:
    """
    Return Dice Loss module suitable for binary segmentation under high class imbalance.
    """
    return smp.losses.DiceLoss(mode=mode, from_logits=from_logits)


def save_model_weights(model: nn.Module, model_path: str) -> None:
    """
    Export the model state dictionary to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_model_weights(
    model: nn.Module,
    model_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load state dictionary into existing model architecture.
    """
    if device is None:
        device = get_default_device()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class SARFloodSegmenter:
    """
    Production-grade inference pipeline encapsulating the SAR U-Net model.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()
        self.model = model or build_unet_model(device=self.device)

        if weights_path and os.path.exists(weights_path):
            load_model_weights(self.model, weights_path, self.device)
        self.model.eval()

    def train_epochs(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
        loss_fn: Optional[nn.Module] = None,
    ) -> List[float]:
        """
        Train the model over several epochs and return the epoch loss progression.
        """
        if loss_fn is None:
            loss_fn = get_loss_fn()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(images)
                loss = loss_fn(predictions, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(train_loader))
            loss_history.append(avg_loss)

        self.model.eval()
        return loss_history

    def predict_probabilities(
        self,
        sar_tensor: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """
        Produce pixel-wise sigmoid water probability map in range [0.0, 1.0].
        Expected input shape: (2, H, W) or (1, 2, H, W).
        Returns: (H, W) numpy float32 array.
        """
        self.model.eval()

        if isinstance(sar_tensor, np.ndarray):
            sar_tensor = torch.from_numpy(sar_tensor).float()

        if sar_tensor.ndim == 3:
            # (2, H, W) -> add batch dim (1, 2, H, W)
            sar_tensor = sar_tensor.unsqueeze(0)

        sar_tensor = sar_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(sar_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        return probs.astype(np.float32)

    def predict_mask(
        self,
        sar_tensor: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Produce binary flood mask array of shape (H, W), values {0.0, 1.0}.
        """
        probs = self.predict_probabilities(sar_tensor)
        return (probs > threshold).astype(np.float32)
