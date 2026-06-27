"""Atrous Spatial Pyramid Pooling (ASPP) bridge module."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _conv_bn_act(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 1,
    dilation_rate: int = 1,
    l2_weight: float = 1e-4,
    name: str = "",
) -> tf.Tensor:
    """Conv2D → BatchNorm → Swish with He-normal init and L2 regularization."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2_weight),
        use_bias=False,
        name=f"{name}_conv" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.Activation("swish", name=f"{name}_swish" if name else None)(x)
    return x


def aspp_block(
    x: tf.Tensor,
    filters: int = 256,
    rates: Tuple[int, ...] = (6, 12, 18),
    l2_weight: float = 1e-4,
    name: str = "aspp",
) -> tf.Tensor:
    """
    Build an ASPP module with parallel atrous convolutions and global pooling.

    Args:
        x: Input feature map from the encoder bottleneck.
        filters: Number of filters in each ASPP branch.
        rates: Dilation rates for atrous convolutions.
        l2_weight: L2 regularization weight.
        name: Layer name prefix.

    Returns:
        Refined feature map after ASPP processing.
    """
    branches = []

    # 1×1 convolution branch
    branches.append(_conv_bn_act(x, filters, kernel_size=1, l2_weight=l2_weight, name=f"{name}_1x1"))

    # Parallel atrous convolutions
    for i, rate in enumerate(rates):
        branches.append(
            _conv_bn_act(
                x,
                filters,
                kernel_size=3,
                dilation_rate=rate,
                l2_weight=l2_weight,
                name=f"{name}_rate{rate}",
            )
        )

    # Global average pooling branch
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    gap = layers.Reshape((1, 1, -1), name=f"{name}_gap_reshape")(gap)
    gap = _conv_bn_act(gap, filters, kernel_size=1, l2_weight=l2_weight, name=f"{name}_gap_conv")
    target_h, target_w = x.shape[1], x.shape[2]
    gap = layers.UpSampling2D(
        size=(target_h, target_w),
        interpolation="bilinear",
        name=f"{name}_gap_upsample",
    )(gap)
    branches.append(gap)

    # Concatenate all branches
    x = layers.Concatenate(name=f"{name}_concat")(branches)
    x = _conv_bn_act(x, filters, kernel_size=1, l2_weight=l2_weight, name=f"{name}_project")
    x = layers.SpatialDropout2D(0.1, name=f"{name}_dropout")(x)

    return x
