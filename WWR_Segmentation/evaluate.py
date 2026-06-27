"""Evaluation on validation and independent test sets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tensorflow import keras

from WWR_Segmentation.config import Config
from WWR_Segmentation.dataset import get_test_dataset, get_val_dataset
from WWR_Segmentation.metrics import compute_per_class_metrics_from_cm
from WWR_Segmentation.utils import prepare_environment

logger = logging.getLogger("wwr_segmentation")


def _predict_dataset(
    model: keras.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a tf.data dataset and return (y_true, y_pred) label arrays."""
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []

    for images, masks in dataset:
        preds = model.predict(images, verbose=0)
        y_pred = np.argmax(preds, axis=-1)
        y_true = np.argmax(masks.numpy(), axis=-1) if masks.shape[-1] > 1 else masks.numpy()
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    return np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)


def evaluate_split(
    model: keras.Model,
    config: Config,
    split: str = "val",
    output_subdir: Optional[str] = None,
) -> Dict:
    """
    Evaluate *model* on validation or independent test split.

    Args:
        model: Trained segmentation model.
        config: Project configuration.
        split: ``'val'`` or ``'test'``.
        output_subdir: Optional subdirectory under predictions_dir for outputs.

    Returns:
        Dictionary with metrics, confusion matrix, and output paths.
    """
    if split == "test":
        dataset = get_test_dataset(config, one_hot=True)
    elif split == "val":
        dataset = get_val_dataset(config, one_hot=True)
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'val' or 'test'.")

    logger.info("Evaluating on %s split...", split)
    y_true, y_pred = _predict_dataset(model, dataset)

    cm = sk_confusion_matrix(
        y_true.ravel(), y_pred.ravel(), labels=list(range(config.num_classes))
    )
    metrics = compute_per_class_metrics_from_cm(cm, config.class_names)

    out_dir = config.predictions_dir / (output_subdir or split)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_confusion_matrix(cm, config.class_names, out_dir / "confusion_matrix.png")
    _save_metrics_json(metrics, out_dir / "metrics.json")
    generate_prediction_gallery(model, config, dataset, out_dir / "gallery", num_samples=8)
    generate_failure_cases(y_true, y_pred, config, out_dir / "failures", num_cases=6)

    logger.info("%s — mean IoU: %.4f, pixel accuracy: %.4f",
                split.capitalize(),
                metrics["mean_iou"]["value"],
                metrics["pixel_accuracy"]["value"])

    return {"metrics": metrics, "confusion_matrix": cm.tolist(), "output_dir": str(out_dir)}


def evaluate_all(
    model: keras.Model,
    config: Config,
) -> Dict[str, Dict]:
    """Evaluate on both validation and independent test sets."""
    results = {
        "validation": evaluate_split(model, config, split="val"),
        "test": evaluate_split(model, config, split="test"),
    }
    summary_path = config.output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v["metrics"] for k, v in results.items()},
            f,
            indent=2,
        )
    logger.info("Evaluation summary saved to %s", summary_path)
    return results


def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: Tuple[str, ...],
    save_path: Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_metrics_json(metrics: Dict, save_path: Path) -> None:
    """Save per-class metrics to JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _colorize_mask(mask: np.ndarray, config: Config) -> np.ndarray:
    """Convert integer label mask to RGB visualization."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(config.class_colors):
        colored[mask == class_idx] = color
    return colored


def generate_prediction_gallery(
    model: keras.Model,
    config: Config,
    dataset: tf.data.Dataset,
    save_dir: Path,
    num_samples: int = 8,
) -> None:
    """Save side-by-side image / ground-truth / prediction overlays."""
    save_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (images, masks) in enumerate(dataset.take(num_samples)):
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=-1)
        true_labels = np.argmax(masks.numpy(), axis=-1)

        for i in range(min(images.shape[0], 1)):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            img = images[i].numpy()
            axes[0].imshow(img)
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(_colorize_mask(true_labels[i], config))
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(_colorize_mask(pred_labels[i], config))
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            overlay = img.copy()
            pred_colored = _colorize_mask(pred_labels[i], config).astype(np.float32) / 255.0
            axes[3].imshow(0.6 * img + 0.4 * pred_colored)
            axes[3].set_title("Overlay")
            axes[3].axis("off")

            fig.savefig(save_dir / f"sample_{batch_idx:03d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


def generate_failure_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Config,
    save_dir: Path,
    num_cases: int = 6,
) -> None:
    """Identify and visualize images with the lowest per-image IoU."""
    save_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    for i in range(y_true.shape[0]):
        intersection = np.sum(y_true[i] == y_pred[i])
        union = np.sum(y_true[i] != 255) + np.sum(y_pred[i] != 255) - intersection
        iou = intersection / (union + 1e-6)
        ious.append((i, iou))

    ious.sort(key=lambda x: x[1])
    worst = ious[:num_cases]

    for rank, (idx, iou) in enumerate(worst):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(_colorize_mask(y_true[idx], config))
        axes[0].set_title(f"Ground Truth (IoU={iou:.3f})")
        axes[0].axis("off")
        axes[1].imshow(_colorize_mask(y_pred[idx], config))
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        fig.savefig(save_dir / f"failure_{rank:02d}_iou{iou:.3f}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def run_evaluation(config: Config, model_path: Optional[str] = None) -> Dict[str, Dict]:
    """Load best model and run full evaluation pipeline."""
    prepare_environment(config)
    path = model_path or str(config.best_model_path)

    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")

    logger.info("Loading model from %s", path)
    model = keras.models.load_model(path, compile=False)
    return evaluate_all(model, config)
