"""Optional 5-fold cross-validation without touching the independent test set."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tensorflow import keras

from WWR_Segmentation.callbacks import get_callbacks
from WWR_Segmentation.config import Config
from WWR_Segmentation.dataset import get_cv_fold_pairs, get_dataset_info, make_dataset_from_pair_list
from WWR_Segmentation.losses import get_loss
from WWR_Segmentation.metrics import get_keras_metrics
from WWR_Segmentation.model import build_segmentation_model
from WWR_Segmentation.optimizer import build_optimizer
from WWR_Segmentation.utils import prepare_environment

logger = logging.getLogger("wwr_segmentation")


def run_cross_validation(config: Config) -> Dict:
    """
    Run K-fold cross-validation on the training set only.

    The independent test set is never included in any fold.

    Args:
        config: Project configuration.

    Returns:
        Dictionary with per-fold and aggregated metrics.
    """
    prepare_environment(config)
    num_folds = config.cv_folds

    fold_results: List[Dict] = []

    for fold in range(num_folds):
        logger.info("=" * 60)
        logger.info("Cross-validation fold %d / %d", fold + 1, num_folds)
        logger.info("=" * 60)

        train_pairs, val_pairs = get_cv_fold_pairs(config, fold, num_folds)
        logger.info("Fold %d — train: %d, val: %d", fold + 1, len(train_pairs), len(val_pairs))

        train_ds = make_dataset_from_pair_list(train_pairs, config, training=True, one_hot=True)
        val_ds = make_dataset_from_pair_list(val_pairs, config, training=False, one_hot=True)

        steps_per_epoch = max(len(train_pairs) // config.batch_size, 1)
        validation_steps = max(len(val_pairs) // config.batch_size, 1)

        model = build_segmentation_model(config)
        optimizer = build_optimizer(config, steps_per_epoch)
        model.compile(
            optimizer=optimizer,
            loss=get_loss(config),
            metrics=get_keras_metrics(config),
            jit_compile=config.xla_jit,
        )

        run_name = f"cv_fold_{fold + 1}"
        callbacks = get_callbacks(config, run_name=run_name)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.total_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        best_val_iou = float(max(history.history.get("val_mean_iou", [0.0])))
        fold_results.append({
            "fold": fold + 1,
            "train_samples": len(train_pairs),
            "val_samples": len(val_pairs),
            "best_val_mean_iou": best_val_iou,
            "epochs_trained": len(history.history.get("loss", [])),
        })

        fold_model_path = config.checkpoint_dir / f"cv_fold_{fold + 1}_best.keras"
        model.save(fold_model_path)
        logger.info("Fold %d best val_mean_iou: %.4f — saved to %s",
                    fold + 1, best_val_iou, fold_model_path)

    iou_scores = [f["best_val_mean_iou"] for f in fold_results]
    summary = {
        "num_folds": num_folds,
        "fold_results": fold_results,
        "mean_val_iou": float(np.mean(iou_scores)),
        "std_val_iou": float(np.std(iou_scores)),
        "test_set_used": False,
    }

    summary_path = config.output_dir / "cross_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Cross-validation complete — mean val IoU: %.4f ± %.4f",
                summary["mean_val_iou"], summary["std_val_iou"])
    logger.info("Summary saved to %s", summary_path)

    return summary
