"""tf.data pipeline for loading, preprocessing, and splitting the dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from WWR_Segmentation.augmentation import get_augment_fn
from WWR_Segmentation.config import Config
from WWR_Segmentation.utils import list_image_files, pair_images_with_masks, set_random_seeds


def _decode_image(path: tf.Tensor) -> tf.Tensor:
    """Load and decode an RGB image from disk."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.cast(image, tf.float32) / 255.0


def _decode_mask(path: tf.Tensor, lut: tf.Tensor) -> tf.Tensor:
    """Load a grayscale mask and remap pixel values to class indices via LUT."""
    mask_bytes = tf.io.read_file(path)
    mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask.set_shape([None, None, 1])
    mask = tf.cast(mask[..., 0], tf.int32)
    return tf.gather(lut, tf.clip_by_value(mask, 0, 255))


def _resize_pair(
    image: tf.Tensor,
    mask: tf.Tensor,
    target_size: Tuple[int, int],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Resize image (bilinear) and mask (nearest neighbor)."""
    image = tf.image.resize(image, target_size, method="bilinear")
    mask = tf.image.resize(
        tf.expand_dims(tf.cast(mask, tf.float32), -1),
        target_size,
        method="nearest",
    )
    mask = tf.cast(tf.squeeze(mask, -1), tf.int32)
    return image, mask


def _one_hot_encode(mask: tf.Tensor, num_classes: int) -> tf.Tensor:
    """Convert integer mask to one-hot encoding."""
    return tf.one_hot(mask, depth=num_classes, dtype=tf.float32)


def _load_sample(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    lut: tf.Tensor,
    image_size: Tuple[int, int],
    num_classes: int,
    one_hot: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load, resize, and optionally one-hot encode a single sample."""
    image = _decode_image(image_path)
    mask = _decode_mask(mask_path, lut)
    image, mask = _resize_pair(image, mask, image_size)

    if one_hot:
        mask = _one_hot_encode(mask, num_classes)
    else:
        mask = tf.cast(mask, tf.int32)

    return image, mask


def has_test_set(config: Config) -> bool:
    """True if the independent test split exists and contains paired images/masks."""
    img_dir = Path(config.test_images_dir)
    msk_dir = Path(config.test_masks_dir)
    if not img_dir.exists() or not msk_dir.exists():
        return False
    images = list_image_files(img_dir, config.image_extensions, required=False)
    return len(images) > 0


def build_file_pairs(config: Config, split: str = "train") -> List[Tuple[str, str]]:
    """
    Build (image, mask) path pairs for the requested split.

    Args:
        config: Project configuration.
        split: One of ``'train'``, ``'val'``, ``'test'``.
    """
    if split == "test":
        if not has_test_set(config):
            return []
        image_paths = list_image_files(config.test_images_dir, config.image_extensions)
        return pair_images_with_masks(image_paths, config.test_masks_dir, config.image_extensions)

    image_paths = list_image_files(config.train_images_dir, config.image_extensions)
    all_pairs = pair_images_with_masks(
        image_paths, config.train_masks_dir, config.image_extensions
    )

    set_random_seeds(config.seed)
    indices = np.arange(len(all_pairs))
    np.random.shuffle(indices)

    val_count = int(len(all_pairs) * config.val_split)
    val_indices = set(indices[:val_count].tolist())

    if split == "val":
        return [all_pairs[i] for i in sorted(val_indices)]
    return [all_pairs[i] for i in range(len(all_pairs)) if i not in val_indices]


def _make_dataset_from_pairs(
    pairs: List[Tuple[str, str]],
    config: Config,
    training: bool = False,
    one_hot: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from (image, mask) path pairs."""
    image_paths, mask_paths = zip(*pairs) if pairs else ([], [])
    lut = tf.constant(config.mask_lut, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))

    load_fn = lambda img_p, msk_p: _load_sample(  # noqa: E731
        img_p, msk_p, lut, config.image_size, config.num_classes, one_hot
    )

    dataset = dataset.map(
        load_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    if training:
        dataset = dataset.shuffle(config.shuffle_buffer, seed=config.seed)
        augment_fn = get_augment_fn(config)
        dataset = dataset.map(
            augment_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(config.batch_size)
    prefetch = config.prefetch_buffer if config.prefetch_buffer is not None else tf.data.AUTOTUNE
    dataset = dataset.prefetch(prefetch)
    return dataset


def get_train_dataset(config: Config, one_hot: bool = True) -> tf.data.Dataset:
    """Return the training tf.data pipeline (with augmentation and caching)."""
    pairs = build_file_pairs(config, split="train")
    dataset = _make_dataset_from_pairs(pairs, config, training=True, one_hot=one_hot)
    return dataset.cache()


def get_val_dataset(config: Config, one_hot: bool = True) -> tf.data.Dataset:
    """Return the validation tf.data pipeline."""
    pairs = build_file_pairs(config, split="val")
    return _make_dataset_from_pairs(pairs, config, training=False, one_hot=one_hot)


def get_test_dataset(config: Config, one_hot: bool = True) -> tf.data.Dataset:
    """Return the independent test tf.data pipeline (never used during training)."""
    pairs = build_file_pairs(config, split="test")
    if not pairs:
        raise FileNotFoundError(
            "Independent test set not available. "
            "Upload test data to Drive when ready, then re-run setup with force_unzip=True."
        )
    return _make_dataset_from_pairs(pairs, config, training=False, one_hot=one_hot)


def get_dataset_info(config: Config) -> Dict[str, int]:
    """Return sample counts for each split."""
    info = {
        "train": len(build_file_pairs(config, "train")),
        "val": len(build_file_pairs(config, "val")),
        "test": len(build_file_pairs(config, "test")) if has_test_set(config) else 0,
    }
    return info


def get_cv_fold_pairs(
    config: Config, fold_index: int, num_folds: int = 5
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Return train/validation pairs for a specific cross-validation fold.

    The independent test set is never included.
    """
    image_paths = list_image_files(config.train_images_dir, config.image_extensions)
    all_pairs = pair_images_with_masks(
        image_paths, config.train_masks_dir, config.image_extensions
    )

    set_random_seeds(config.seed)
    indices = np.arange(len(all_pairs))
    np.random.shuffle(indices)
    shuffled_pairs = [all_pairs[i] for i in indices]

    fold_size = len(shuffled_pairs) // num_folds
    val_start = fold_index * fold_size
    val_end = val_start + fold_size if fold_index < num_folds - 1 else len(shuffled_pairs)

    val_pairs = shuffled_pairs[val_start:val_end]
    train_pairs = shuffled_pairs[:val_start] + shuffled_pairs[val_end:]
    return train_pairs, val_pairs


def make_dataset_from_pair_list(
    pairs: List[Tuple[str, str]],
    config: Config,
    training: bool = False,
    one_hot: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from an explicit list of path pairs."""
    dataset = _make_dataset_from_pairs(pairs, config, training=training, one_hot=one_hot)
    if training:
        dataset = dataset.cache()
    return dataset
