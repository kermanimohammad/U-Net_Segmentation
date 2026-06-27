"""Residual decoder with progressive upsampling and skip fusion."""

from __future__ import annotations

from typing import List, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _match_channels(x: tf.Tensor, target_channels: int, l2_weight: float, name: str) -> tf.Tensor:
    """Project feature map to *target_channels* via 1×1 convolution."""
    return layers.Conv2D(
        target_channels,
        1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2_weight),
        use_bias=False,
        name=f"{name}_channel_match",
    )(x)


def _align_spatial(
    x: tf.Tensor, reference: tf.Tensor, name: str
) -> tf.Tensor:
    """Resize *x* to match the spatial dimensions of *reference*."""
    return layers.Resizing(
        reference.shape[1],
        reference.shape[2],
        interpolation="bilinear",
        name=f"{name}_resize",
    )(x)


def residual_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.3,
    l2_weight: float = 1e-4,
    name: str = "res_block",
) -> tf.Tensor:
    """
    Residual block: Conv-BN-Swish → Conv-BN → Add → Swish → SpatialDropout.

    Uses He-normal initialization and L2 regularization throughout.
    """
    shortcut = x

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2_weight),
            use_bias=False,
            name=f"{name}_shortcut",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2_weight),
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("swish", name=f"{name}_swish1")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2_weight),
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("swish", name=f"{name}_swish2")(x)
    x = layers.SpatialDropout2D(dropout_rate, name=f"{name}_dropout")(x)

    return x


def decoder_block(
    x: tf.Tensor,
    skip: Optional[tf.Tensor],
    filters: int,
    dropout_rate: float = 0.3,
    l2_weight: float = 1e-4,
    name: str = "decoder_block",
) -> tf.Tensor:
    """
    Upsample, fuse skip connection, and apply a residual block.

    Args:
        x: Feature map from the previous decoder stage.
        skip: Encoder skip connection (or None for the deepest stage).
        filters: Output filter count.
        dropout_rate: SpatialDropout rate.
        l2_weight: L2 regularization weight.
        name: Layer name prefix.
    """
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_upsample")(x)

    if skip is not None:
        skip = _match_channels(skip, filters, l2_weight, f"{name}_skip")
        skip = _align_spatial(skip, x, f"{name}_skip")
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])

    x = residual_block(x, filters, dropout_rate, l2_weight, name=f"{name}_res")
    return x


def build_decoder(
    bridge: tf.Tensor,
    skip_connections: List[tf.Tensor],
    decoder_filters: tuple[int, ...] = (256, 128, 64, 32),
    dropout_rate: float = 0.3,
    l2_weight: float = 1e-4,
    name: str = "decoder",
) -> tf.Tensor:
    """
    Build the full residual decoder with progressive upsampling.

    Args:
        bridge: ASPP output (deepest feature map).
        skip_connections: Encoder skip features ordered from deepest to shallowest.
        decoder_filters: Filter counts per decoder stage.
        dropout_rate: SpatialDropout rate.
        l2_weight: L2 regularization weight.
        name: Layer name prefix.

    Returns:
        Decoder output feature map at 1/4 input resolution (128×128 for 512 input).
    """
    x = bridge
    num_stages = len(decoder_filters)

    for i, filters in enumerate(decoder_filters):
        skip_idx = num_stages - 1 - i
        skip = skip_connections[skip_idx] if skip_idx < len(skip_connections) else None
        x = decoder_block(
            x,
            skip,
            filters,
            dropout_rate=dropout_rate,
            l2_weight=l2_weight,
            name=f"{name}_stage{i}",
        )

    return x
