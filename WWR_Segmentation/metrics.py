"""Segmentation evaluation metrics."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config


def _one_hot_if_needed(y: tf.Tensor, num_classes: int) -> tf.Tensor:
    """Convert integer labels to one-hot if not already one-hot."""
    if y.shape.rank == 4 and y.shape[-1] == num_classes:
        return tf.cast(y, tf.float32)
    return tf.one_hot(tf.cast(y, tf.int32), depth=num_classes, dtype=tf.float32)


def _argmax_predictions(y_pred: tf.Tensor) -> tf.Tensor:
    """Convert softmax output to hard class predictions."""
    return tf.argmax(y_pred, axis=-1, output_type=tf.int32)


def pixel_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Fraction of correctly classified pixels."""
    y_true_int = tf.argmax(y_true, axis=-1, output_type=tf.int32) if y_true.shape.rank == 4 else tf.cast(y_true, tf.int32)
    y_pred_int = _argmax_predictions(y_pred)
    correct = tf.cast(tf.equal(y_true_int, y_pred_int), tf.float32)
    return tf.reduce_mean(correct)


def _per_class_iou(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int, smooth: float = 1e-6
) -> tf.Tensor:
    """Compute IoU for each class; returns shape (num_classes,)."""
    y_true_int = tf.argmax(y_true, axis=-1) if y_true.shape.rank == 4 else tf.cast(y_true, tf.int32)
    y_pred_int = _argmax_predictions(y_pred)

    ious = []
    for c in range(num_classes):
        true_c = tf.cast(tf.equal(y_true_int, c), tf.float32)
        pred_c = tf.cast(tf.equal(y_pred_int, c), tf.float32)
        intersection = tf.reduce_sum(true_c * pred_c)
        union = tf.reduce_sum(true_c) + tf.reduce_sum(pred_c) - intersection
        ious.append((intersection + smooth) / (union + smooth))

    return tf.stack(ious)


def mean_iou(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 4) -> tf.Tensor:
    """Mean Intersection over Union across all classes."""
    return tf.reduce_mean(_per_class_iou(y_true, y_pred, num_classes))


def _per_class_dice(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int, smooth: float = 1e-6
) -> tf.Tensor:
    """Compute Dice coefficient for each class."""
    y_true_oh = _one_hot_if_needed(y_true, num_classes)
    y_pred = tf.cast(y_pred, tf.float32)

    dices = []
    for c in range(num_classes):
        t = y_true_oh[..., c]
        p = y_pred[..., c]
        intersection = tf.reduce_sum(t * p)
        dices.append(
            (2.0 * intersection + smooth) / (tf.reduce_sum(t) + tf.reduce_sum(p) + smooth)
        )
    return tf.stack(dices)


def mean_dice(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int = 4) -> tf.Tensor:
    """Mean Dice score across all classes."""
    return tf.reduce_mean(_per_class_dice(y_true, y_pred, num_classes))


def _per_class_precision_recall_f1(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int, smooth: float = 1e-6
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute per-class precision, recall, and F1."""
    y_true_int = tf.argmax(y_true, axis=-1) if y_true.shape.rank == 4 else tf.cast(y_true, tf.int32)
    y_pred_int = _argmax_predictions(y_pred)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        true_c = tf.cast(tf.equal(y_true_int, c), tf.float32)
        pred_c = tf.cast(tf.equal(y_pred_int, c), tf.float32)
        tp = tf.reduce_sum(true_c * pred_c)
        fp = tf.reduce_sum(pred_c) - tp
        fn = tf.reduce_sum(true_c) - tp

        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        f1 = 2.0 * precision * recall / (precision + recall + smooth)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return tf.stack(precisions), tf.stack(recalls), tf.stack(f1s)


def get_keras_metrics(config: Config) -> List[keras.metrics.Metric]:
    """Return the list of Keras metrics for model compilation."""
    nc = config.num_classes

    metrics: List[keras.metrics.Metric] = [
        keras.metrics.MeanMetricWrapper(
            lambda yt, yp: pixel_accuracy(yt, yp), name="pixel_accuracy"
        ),
        keras.metrics.MeanMetricWrapper(
            lambda yt, yp: mean_iou(yt, yp, nc), name="mean_iou"
        ),
        keras.metrics.MeanMetricWrapper(
            lambda yt, yp: mean_dice(yt, yp, nc), name="mean_dice"
        ),
    ]

    for i, class_name in enumerate(config.class_names):
        class_idx = i

        def make_iou_metric(idx: int = class_idx) -> Callable:
            return lambda yt, yp: _per_class_iou(yt, yp, nc)[idx]

        def make_dice_metric(idx: int = class_idx) -> Callable:
            return lambda yt, yp: _per_class_dice(yt, yp, nc)[idx]

        def make_precision_metric(idx: int = class_idx) -> Callable:
            return lambda yt, yp: _per_class_precision_recall_f1(yt, yp, nc)[0][idx]

        def make_recall_metric(idx: int = class_idx) -> Callable:
            return lambda yt, yp: _per_class_precision_recall_f1(yt, yp, nc)[1][idx]

        def make_f1_metric(idx: int = class_idx) -> Callable:
            return lambda yt, yp: _per_class_precision_recall_f1(yt, yp, nc)[2][idx]

        metrics.extend([
            keras.metrics.MeanMetricWrapper(make_iou_metric(), name=f"iou_{class_name.lower()}"),
            keras.metrics.MeanMetricWrapper(make_dice_metric(), name=f"dice_{class_name.lower()}"),
            keras.metrics.MeanMetricWrapper(make_precision_metric(), name=f"precision_{class_name.lower()}"),
            keras.metrics.MeanMetricWrapper(make_recall_metric(), name=f"recall_{class_name.lower()}"),
            keras.metrics.MeanMetricWrapper(make_f1_metric(), name=f"f1_{class_name.lower()}"),
        ])

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """
    Compute confusion matrix from integer label arrays.

    Args:
        y_true: Ground truth labels (N, H, W) or flat.
        y_pred: Predicted labels (N, H, W) or flat.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_flat, y_pred_flat):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_per_class_metrics_from_cm(
    cm: np.ndarray, class_names: tuple[str, ...]
) -> Dict[str, Dict[str, float]]:
    """Derive per-class IoU, precision, recall, F1, and Dice from a confusion matrix."""
    results: Dict[str, Dict[str, float]] = {}
    num_classes = cm.shape[0]

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)

        results[name] = {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "dice": float(dice),
            "support": int(cm[i, :].sum()),
        }

    results["mean_iou"] = {"value": float(np.mean([results[n]["iou"] for n in class_names]))}
    results["mean_dice"] = {"value": float(np.mean([results[n]["dice"] for n in class_names]))}
    results["pixel_accuracy"] = {"value": float(np.trace(cm) / (cm.sum() + 1e-6))}

    return results
