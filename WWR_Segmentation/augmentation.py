"""Realistic data augmentation for facade segmentation."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from WWR_Segmentation.config import Config


def _apply_horizontal_flip(
    image: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply horizontal flip to image and mask jointly."""
    return tf.image.random_flip_left_right(image), tf.image.random_flip_left_right(
        tf.expand_dims(mask, -1)
    )[..., 0]


def _apply_color_jitter(image: tf.Tensor, config: Config) -> tf.Tensor:
    """Apply brightness, contrast, and saturation adjustments (image only)."""
    image = tf.image.random_brightness(image, max_delta=config.brightness_delta)
    image = tf.image.random_contrast(
        image, lower=config.contrast_range[0], upper=config.contrast_range[1]
    )
    image = tf.image.random_saturation(
        image, lower=config.saturation_range[0], upper=config.saturation_range[1]
    )
    return tf.clip_by_value(image, 0.0, 1.0)


def _apply_gaussian_noise(image: tf.Tensor, std: float) -> tf.Tensor:
    """Add small Gaussian noise to the image."""
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=std, dtype=image.dtype)
    return tf.clip_by_value(image + noise, 0.0, 1.0)


def augment(image: tf.Tensor, mask: tf.Tensor, config: Config) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply realistic augmentations that preserve fine window structures.

    Allowed: horizontal flip, brightness, contrast, saturation, small Gaussian noise.
    Forbidden: vertical flip, rotation, perspective, large zoom, morphological ops.
    """
    if tf.random.uniform([]) > config.augment_probability:
        return image, mask

    if config.horizontal_flip:
        do_flip = tf.random.uniform([]) > 0.5
        image, mask = tf.cond(
            do_flip,
            lambda: _apply_horizontal_flip(image, mask),
            lambda: (image, mask),
        )

    image = _apply_color_jitter(image, config)
    image = _apply_gaussian_noise(image, config.gaussian_noise_std)

    return image, mask


def get_augment_fn(config: Config):
    """Return a closure suitable for tf.data.Dataset.map()."""
    def _augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return augment(image, mask, config)

    return _augment
