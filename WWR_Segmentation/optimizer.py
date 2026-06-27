"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras

from WWR_Segmentation.config import Config


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine annealing to *min_lr*."""

    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))
        warmup_lr = tf.clip_by_value(warmup_lr, 0.0, self.initial_lr)

        # Cosine decay after warmup
        decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value((step - warmup_steps) / decay_steps, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> dict:
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


def build_optimizer(config: Config, steps_per_epoch: int) -> keras.optimizers.Optimizer:
    """
    Build AdamW optimizer with warmup-cosine LR, weight decay, and gradient clipping.

    Args:
        config: Project configuration.
        steps_per_epoch: Training steps per epoch (for LR schedule).
    """
    total_steps = steps_per_epoch * config.total_epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    lr_schedule = WarmupCosineDecay(
        initial_lr=config.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_learning_rate,
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=config.gradient_clip_norm,
        use_ema=False,
    )

    if config.mixed_precision:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer
