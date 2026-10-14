"""
Unit tests for FloodLens AI core data, neural model, metrics, and routing pipeline.
"""

import pytest
import torch
import numpy as np
from shapely.geometry import box

from src.dataset import SyntheticSARFloodDataset, get_dataloaders
from src.model import build_unet_model, get_loss_fn, SARFloodSegmenter
from src.evaluate import compute_binary_metrics
from src.router import RoadNetworkRouter, DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


def test_synthetic_sar_dataset_shapes_and_normalization():
    """Verify SAR image tensor shapes, channels, value ranges, and mask labels."""
    dataset = SyntheticSARFloodDataset(num_samples=4, img_size=128, seed=42)
    assert len(dataset) == 4

    sar, mask = dataset[0]
    # Check tensor dimensions
    assert sar.shape == (2, 128, 128)
    assert mask.shape == (1, 128, 128)

    # Check physical normalization range [0, 1]
    assert float(sar.min()) >= 0.0
    assert float(sar.max()) <= 1.0

    # Check ground truth mask binary values
    unique_vals = torch.unique(mask).numpy()
    for val in unique_vals:
        assert val in [0.0, 1.0]

    # Ground truth must contain flooded pixels
    assert (mask == 1.0).sum() > 0


def test_dataset_dataloader_batching():
    """Verify PyTorch DataLoader batching and shapes."""
    train_loader, val_loader = get_dataloaders(
        num_train=8, num_val=4, batch_size=4, img_size=128, seed=123
    )
    sar_batch, mask_batch = next(iter(train_loader))

    assert sar_batch.shape == (4, 2, 128, 128)
    assert mask_batch.shape == (4, 1, 128, 128)
    assert sar_batch.dtype == torch.float32
    assert mask_batch.dtype == torch.float32


def test_unet_architecture_and_loss():
    """Verify U-Net forward pass, output shape, and Dice loss calculation."""
    device = torch.device("cpu")
    model = build_unet_model(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        device=device,
    )

    batch_x = torch.rand(2, 2, 128, 128, device=device)
    batch_y = torch.randint(0, 2, (2, 1, 128, 128), device=device).float()

    logits = model(batch_x)
    assert logits.shape == (2, 1, 128, 128)

    loss_fn = get_loss_fn()
    loss = loss_fn(logits, batch_y)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_sar_flood_segmenter_prediction():
    """Verify inference pipeline wrapper predicts valid probability maps and binary masks."""
    segmenter = SARFloodSegmenter(device=torch.device("cpu"))
    dummy_input = np.random.uniform(0.0, 1.0, (2, 128, 128)).astype(np.float32)

    probs = segmenter.predict_probabilities(dummy_input)
    assert probs.shape == (128, 128)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    mask = segmenter.predict_mask(dummy_input, threshold=0.5)
    assert mask.shape == (128, 128)
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_evaluation_binary_metrics():
    """Verify IoU, Dice, Precision, and Recall calculation logic."""
    # Test perfect prediction
    y_true = np.array([[1, 1, 0], [0, 1, 0]])
    y_pred = np.array([[1, 1, 0], [0, 1, 0]])

    metrics = compute_binary_metrics(y_true, y_pred)
    assert metrics["tp"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 3
    assert pytest.approx(metrics["iou"], 0.001) == 1.0
    assert pytest.approx(metrics["dice"], 0.001) == 1.0
    assert pytest.approx(metrics["precision"], 0.001) == 1.0
    assert pytest.approx(metrics["recall"], 0.001) == 1.0


def test_router_graph_severing_and_dijkstra():
    """Verify road graph ingestion, edge severing, and safe path calculation."""
    router = RoadNetworkRouter(
        center_lat=DEFAULT_CENTER_LAT,
        center_lon=DEFAULT_CENTER_LON,
        radius_m=2000,
        auto_load=True,
    )
    assert router.G is not None
    assert len(router.G.nodes) > 0
    assert len(router.G.edges) > 0

    # Sever edges
    poly = router.get_default_flood_polygon()
    g_safe, severed = router.sever_flooded_edges(poly)

    assert len(severed) >= 0
    assert len(g_safe.edges) <= len(router.G.edges)

    # Test route finding
    orig_lat = DEFAULT_CENTER_LAT + 0.005
    orig_lon = DEFAULT_CENTER_LON - 0.005
    dest_lat = DEFAULT_CENTER_LAT - 0.005
    dest_lon = DEFAULT_CENTER_LON + 0.005

    result = router.find_evacuation_route(
        origin_lat=orig_lat,
        origin_lon=orig_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        flood_polygon=poly,
    )

    assert result["status"] == "success"
    assert result["distance_meters"] > 0
    assert len(result["route_coords"]) >= 2
    assert "geojson" in result
    geojson = result["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 3
