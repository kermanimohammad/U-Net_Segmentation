"""Training orchestration for the WWR segmentation model."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.callbacks import get_callbacks
from WWR_Segmentation.config import Config
from WWR_Segmentation.dataset import get_dataset_info, get_train_dataset, get_val_dataset
from WWR_Segmentation.losses import CombinedSegmentationLoss, get_loss
from WWR_Segmentation.metrics import get_keras_metrics
from WWR_Segmentation.model import build_segmentation_model
from WWR_Segmentation.optimizer import build_optimizer
from WWR_Segmentation.utils import prepare_environment

logger = logging.getLogger("wwr_segmentation")


def compile_model(model: keras.Model, config: Config, steps_per_epoch: int) -> keras.Model:
    """Compile *model* with combined loss, metrics, and AdamW optimizer."""
    optimizer = build_optimizer(config, steps_per_epoch)
    model.compile(
        optimizer=optimizer,
        loss=get_loss(config),
        metrics=get_keras_metrics(config),
        jit_compile=config.xla_jit,
    )
    return model


def train(config: Config, run_name: str = "training") -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Full training pipeline: build model, compile, and fit on train/val splits.

    The independent test set is never used during training.

    Args:
        config: Project configuration.
        run_name: Identifier for logs and checkpoints.

    Returns:
        Trained Keras model with best weights restored (via EarlyStopping).
    """
    prepare_environment(config)

    dataset_info = get_dataset_info(config)
    logger.info("Dataset splits — train: %d, val: %d, test: %d (held out)",
                dataset_info["train"], dataset_info["val"], dataset_info["test"])

    train_ds = get_train_dataset(config, one_hot=True)
    val_ds = get_val_dataset(config, one_hot=True)

    steps_per_epoch = max(dataset_info["train"] // config.batch_size, 1)
    validation_steps = max(dataset_info["val"] // config.batch_size, 1)

    resume_path = None
    if config.resume_from_best_model:
        resume_path = config.resume_checkpoint_path
        if not tf.io.gfile.exists(str(resume_path)):
            logger.warning(
                "Resume checkpoint not found at %s — training from scratch.",
                resume_path,
            )
            resume_path = None

    if resume_path is not None:
        logger.info("Fine-tuning from checkpoint: %s", resume_path)
        model = keras.models.load_model(
            str(resume_path),
            custom_objects={"CombinedSegmentationLoss": CombinedSegmentationLoss},
        )
    else:
        logger.info("Building EfficientNetV2-S segmentation model...")
        model = build_segmentation_model(config)

    model = compile_model(model, config, steps_per_epoch)

    model.summary(print_fn=logger.info)

    callbacks = get_callbacks(config, run_name=run_name)

    logger.info("Starting training for up to %d epochs...", config.total_epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.total_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training complete. Best model saved to %s", config.best_model_path)
    return model, history


def load_and_compile_model(
    config: Config,
    model_path: Optional[str] = None,
    steps_per_epoch: int = 1,
) -> keras.Model:
    """Load a saved model or build a fresh one and compile it."""
    path = model_path or str(config.best_model_path)

    if model_path and tf.io.gfile.exists(path):
        logger.info("Loading model from %s", path)
        model = keras.models.load_model(
            path,
            custom_objects={"CombinedSegmentationLoss": CombinedSegmentationLoss},
        )
    else:
        model = build_segmentation_model(config)

    return compile_model(model, config, steps_per_epoch)
