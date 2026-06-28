"""Modular loss functions: Dice, Focal, Boundary, and combined loss."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config


def dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
    class_weights: Optional[Sequence[float]] = None,
) -> tf.Tensor:
    """Multi-class Dice loss — optionally weighted per class."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    if class_weights is not None:
        weights = tf.constant(class_weights, dtype=tf.float32)
        weighted_dice = tf.reduce_sum(dice_per_class * weights, axis=-1) / tf.reduce_sum(weights)
        return 1.0 - tf.reduce_mean(weighted_dice)

    return 1.0 - tf.reduce_mean(dice_per_class)


def focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    class_alpha: Optional[Sequence[float]] = None,
) -> tf.Tensor:
    """Multi-class focal loss for handling class imbalance."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

    cross_entropy = -y_true * tf.math.log(y_pred)
    if class_alpha is not None:
        alpha_tensor = tf.reshape(
            tf.constant(class_alpha, dtype=tf.float32),
            (1, 1, 1, -1),
        )
        weight = alpha_tensor * y_true * tf.pow(1.0 - y_pred, gamma)
    else:
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def _sobel_edges(mask: tf.Tensor) -> tf.Tensor:
    """Compute Sobel edge magnitude for a single-channel mask."""
    mask = tf.cast(mask, tf.float32)
    if mask.shape.rank == 3:
        mask = tf.expand_dims(mask, -1)

    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

    gx = tf.nn.conv2d(mask, sobel_x, strides=1, padding="SAME")
    gy = tf.nn.conv2d(mask, sobel_y, strides=1, padding="SAME")
    return tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-6)


def boundary_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    num_classes: int = 4,
    theta: float = 3.0,
    class_weights: Optional[Sequence[float]] = None,
) -> tf.Tensor:
    """Boundary-aware loss using Sobel edge maps on one-hot labels."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    boundary_maps = []
    for c in range(num_classes):
        edges = _sobel_edges(y_true[..., c])
        boundary_maps.append(edges)

    boundary = tf.concat(boundary_maps, axis=-1)
    boundary = tf.clip_by_value(boundary, 0.0, 1.0)

    if class_weights is not None:
        weights = tf.reshape(tf.constant(class_weights, dtype=tf.float32), (1, 1, 1, -1))
        boundary = boundary * weights

    ce = -y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
    weighted_ce = ce * (1.0 + theta * boundary)
    return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


class CombinedSegmentationLoss(keras.losses.Loss):
    """Weighted combination: Dice + Focal + Boundary (optional per-class weights)."""

    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._class_weights: Optional[Tuple[float, ...]] = None
        self._class_alpha: Optional[Tuple[float, ...]] = None
        if config.use_class_weights:
            self._class_weights = tuple(config.class_weights)
            self._class_alpha = tuple(config.focal_class_alpha)

    def call(self, y_true, y_pred):
        d_loss = dice_loss(y_true, y_pred, class_weights=self._class_weights)
        f_loss = focal_loss(
            y_true,
            y_pred,
            gamma=self.config.focal_gamma,
            alpha=self.config.focal_alpha,
            class_alpha=self._class_alpha,
        )
        b_loss = boundary_loss(
            y_true,
            y_pred,
            num_classes=self.config.num_classes,
            theta=self.config.boundary_theta,
            class_weights=self._class_weights,
        )
        return (
            self.config.dice_weight * d_loss
            + self.config.focal_weight * f_loss
            + self.config.boundary_weight * b_loss
        )


def combined_loss(config: Config) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Return a callable loss function (for custom training loops)."""
    loss_obj = CombinedSegmentationLoss(config)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return loss_obj(y_true, y_pred)

    return loss_fn


def get_loss(config: Config) -> CombinedSegmentationLoss:
    """Return a Keras-compatible combined loss object."""
    return CombinedSegmentationLoss(config, name="combined_segmentation_loss")
