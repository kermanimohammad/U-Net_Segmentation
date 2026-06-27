"""EfficientNetV2-S encoder with ASPP bridge and residual decoder."""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from WWR_Segmentation.aspp import aspp_block
from WWR_Segmentation.config import Config
from WWR_Segmentation.decoder import build_decoder


# EfficientNetV2-S layer names for skip connections (deepest → shallowest)
EFFICIENTNETV2S_SKIP_LAYERS: Tuple[str, ...] = (
    "block5a_project_bn",
    "block4a_project_bn",
    "block3a_project_bn",
    "block2a_project_bn",
)


def _get_encoder(input_shape: Tuple[int, int, int], weights: str, trainable: bool):
    """Instantiate EfficientNetV2-S without the classification head."""
    base = keras.applications.EfficientNetV2S(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )
    base.trainable = trainable
    return base


def _extract_skip_connections(
    encoder: keras.Model, layer_names: Tuple[str, ...]
) -> List[tf.Tensor]:
    """Extract intermediate feature maps from named encoder layers."""
    outputs = []
    for name in layer_names:
        try:
            layer = encoder.get_layer(name)
        except ValueError as exc:
            available = [l.name for l in encoder.layers if "project_bn" in l.name]
            raise ValueError(
                f"Skip layer '{name}' not found. Available project_bn layers: {available}"
            ) from exc
        outputs.append(layer.output)
    return outputs


def build_segmentation_model(config: Config) -> keras.Model:
    """
    Build the full segmentation model:

    EfficientNetV2-S encoder → ASPP bridge → Residual decoder → Segmentation head.

    Args:
        config: Project configuration.

    Returns:
        Compiled-ready Keras model with softmax output (num_classes channels).
    """
    inputs = layers.Input(shape=(*config.image_size, 3), name="input_image")

    encoder = _get_encoder(
        (*config.image_size, 3),
        weights=config.encoder_weights,
        trainable=config.encoder_trainable,
    )

    # Run encoder and collect skip connections
    skip_tensors = _extract_skip_connections(encoder, EFFICIENTNETV2S_SKIP_LAYERS)
    bottleneck = encoder.output

    # Re-build as a single model for clean forward pass
    encoder_model = keras.Model(
        inputs=encoder.input,
        outputs=[bottleneck] + skip_tensors,
        name="efficientnetv2s_encoder",
    )

    encoder_outputs = encoder_model(inputs)
    bottleneck = encoder_outputs[0]
    skips = list(reversed(encoder_outputs[1:]))  # shallowest last for decoder

    # ASPP bridge
    bridge = aspp_block(
        bottleneck,
        filters=config.aspp_filters,
        rates=config.aspp_rates,
        l2_weight=config.l2_weight,
        name="aspp",
    )

    # Residual decoder with skip fusion
    decoded = build_decoder(
        bridge,
        skip_connections=skips,
        decoder_filters=config.decoder_filters,
        dropout_rate=config.dropout_rate,
        l2_weight=config.l2_weight,
        name="decoder",
    )

    # Final upsampling to full resolution (decoder ends at 128×128 for 512 input)
    x = layers.UpSampling2D(4, interpolation="bilinear", name="final_upsample")(decoded)

    # Segmentation head
    x = layers.Conv2D(
        config.decoder_filters[-1],
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.l2_weight),
        activation="swish",
        name="seg_head_conv",
    )(x)
    x = layers.SpatialDropout2D(config.dropout_rate * 0.5, name="seg_head_dropout")(x)

    outputs = layers.Conv2D(
        config.num_classes,
        1,
        padding="same",
        kernel_initializer="he_normal",
        activation="softmax",
        dtype="float32",
        name="segmentation_output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="wwr_segmentation")
    return model
