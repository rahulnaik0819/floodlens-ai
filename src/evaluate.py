"""
Quantitative Evaluation Benchmark Module for FloodLens AI.
Computes Mean IoU (Jaccard Index), Dice Coefficient (F1), Pixel Precision,
Pixel Recall, and Pixel Breakdown over unseen synthetic SAR scenes.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import SyntheticSARFloodDataset
from src.model import SARFloodSegmenter, build_unet_model, get_default_device


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute binary classification metrics across flattened pixel arrays.
    """
    y_true_bool = y_true.astype(bool).ravel()
    y_pred_bool = y_pred.astype(bool).ravel()

    tp = int(np.logical_and(y_pred_bool, y_true_bool).sum())
    fp = int(np.logical_and(y_pred_bool, ~y_true_bool).sum())
    fn = int(np.logical_and(~y_pred_bool, y_true_bool).sum())
    tn = int(np.logical_and(~y_pred_bool, ~y_true_bool).sum())

    intersection = tp
    union = tp + fp + fn

    iou = float((intersection + eps) / (union + eps))
    dice = float((2 * intersection + eps) / (2 * intersection + fp + fn + eps))
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    pixel_acc = float((tp + tn) / (tp + fp + fn + tn + eps))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": pixel_acc,
    }


def run_benchmark(
    segmenter: SARFloodSegmenter,
    num_test_samples: int = 50,
    img_size: int = 256,
    batch_size: int = 1,
    seed: int = 9999,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Execute quantitative benchmark over unseen test dataset.
    """
    test_dataset = SyntheticSARFloodDataset(
        num_samples=num_test_samples, img_size=img_size, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    iou_scores = []
    dice_scores = []

    dev = device or segmenter.device

    for images, masks in test_loader:
        images = images.to(dev)
        masks = masks.to(dev)

        with torch.no_grad():
            logits = segmenter.model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        y_true = masks.view(-1).cpu().numpy().astype(bool)
        y_pred = preds.view(-1).cpu().numpy().astype(bool)

        tp = np.logical_and(y_pred, y_true).sum()
        fp = np.logical_and(y_pred, ~y_true).sum()
        fn = np.logical_and(~y_pred, y_true).sum()
        tn = np.logical_and(~y_pred, ~y_true).sum()

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        intersection = tp
        union = tp + fp + fn
        batch_iou = (intersection + 1e-6) / (union + 1e-6)
        batch_dice = (2 * intersection + 1e-6) / (2 * intersection + fp + fn + 1e-6)

        iou_scores.append(float(batch_iou))
        dice_scores.append(float(batch_dice))

    mean_iou = float(np.mean(iou_scores))
    mean_dice = float(np.mean(dice_scores))
    precision = float((total_tp + 1e-6) / (total_tp + total_fp + 1e-6))
    recall = float((total_tp + 1e-6) / (total_tp + total_fn + 1e-6))
    pixel_acc = float((total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn))

    results = {
        "num_samples": num_test_samples,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": pixel_acc,
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "total_tn": int(total_tn),
    }

    return results


def print_benchmark_report(metrics: Dict[str, Any]) -> None:
    """
    Format and print the benchmark report.
    """
    print("\n" + "=" * 45)
    print("     FLOODLENS AI EVALUATION BENCHMARK     ")
    print("=" * 45)
    print(f"Test Scenes Evaluated    : {metrics.get('num_samples', 0)}")
    print(f"Mean IoU (Jaccard Index) : {metrics['mean_iou'] * 100:.2f}%")
    print(f"Dice Coefficient (F1)    : {metrics['mean_dice'] * 100:.2f}%")
    print(f"Pixel Precision          : {metrics['precision'] * 100:.2f}%")
    print(f"Pixel Recall (Critical)  : {metrics['recall'] * 100:.2f}%")
    print(f"Raw Pixel Accuracy       : {metrics['pixel_accuracy'] * 100:.2f}%")
    print("=" * 45)
    print(
        f"Pixel Breakdown -> TP: {metrics['total_tp']:,} | "
        f"FP: {metrics['total_fp']:,} | "
        f"FN: {metrics['total_fn']:,} | "
        f"TN: {metrics['total_tn']:,}"
    )
    print("=" * 45 + "\n")


if __name__ == "__main__":
    from src.dataset import get_dataloaders
    print("Instantiating and quickly calibrating model on synthetic data...")
    dev = get_default_device()
    train_loader, _ = get_dataloaders(num_train=64, batch_size=8)
    segmenter = SARFloodSegmenter(device=dev)
    print("Training 3 quick calibration epochs...")
    segmenter.train_epochs(train_loader, epochs=3)
    print("Running quantitative evaluation benchmark on 50 unseen test scenes...")
    results = run_benchmark(segmenter, num_test_samples=50, device=dev)
    print_benchmark_report(results)
