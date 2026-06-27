"""Training callbacks for checkpointing, logging, and early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config


class LearningRateLogger(keras.callbacks.Callback):
    """Log the current learning rate at the end of each epoch."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        if hasattr(self.model.optimizer, "inner_optimizer"):
            lr = float(
                keras.backend.get_value(self.model.optimizer.inner_optimizer.learning_rate)
            )
        if logs is not None:
            logs["learning_rate"] = lr
        print(f"\n  Epoch {epoch + 1} — learning rate: {lr:.2e}")


def get_callbacks(config: Config, run_name: str = "training") -> List[keras.callbacks.Callback]:
    """
    Build the full callback list for training.

    Includes ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard,
    BackupAndRestore, and LearningRateLogger.
    """
    checkpoint_path = config.checkpoint_dir / f"{run_name}_{{epoch:03d}}_{{val_mean_iou:.4f}}.keras"
    best_model_path = config.best_model_path

    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(config.log_dir / f"{run_name}_history.csv"),
            separator=",",
            append=False,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(config.tensorboard_dir / run_name),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=str(config.backup_dir / run_name),
        ),
        LearningRateLogger(),
    ]

    return callbacks
