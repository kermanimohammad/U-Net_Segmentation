"""Central configuration for the WWR segmentation research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Config:
    """All hyperparameters and paths for training, evaluation, and inference."""

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Paths (override for local vs Google Colab / Drive) ───────────────────
    project_root: Path = field(default_factory=lambda: Path("."))
    data_root: Path = field(default_factory=lambda: Path("data"))
    train_images_dir: Path = field(default_factory=lambda: Path("data/train/images"))
    train_masks_dir: Path = field(default_factory=lambda: Path("data/train/masks"))
    test_images_dir: Path = field(default_factory=lambda: Path("data/test/images"))
    test_masks_dir: Path = field(default_factory=lambda: Path("data/test/masks"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("outputs/logs"))
    tensorboard_dir: Path = field(default_factory=lambda: Path("outputs/tensorboard"))
    backup_dir: Path = field(default_factory=lambda: Path("outputs/backup"))
    predictions_dir: Path = field(default_factory=lambda: Path("outputs/predictions"))
    wwr_output_dir: Path = field(default_factory=lambda: Path("outputs/wwr"))

    # Google Colab / Drive
    use_colab: bool = False
    use_google_drive: bool = False
    use_local_data_cache: bool = True
    drive_mount_point: Path = field(default_factory=lambda: Path("/content/drive"))
    drive_project_dir: Path = field(
        default_factory=lambda: Path("/content/drive/MyDrive/WWR_Seg_Model")
    )
    drive_data_zip: Path = field(
        default_factory=lambda: Path("/content/drive/MyDrive/WWR_Seg_Model/data.zip")
    )
    local_data_root: Path = field(default_factory=lambda: Path("/content/data"))
    models_dir: Path = field(default_factory=lambda: Path("outputs/models"))
    results_dir: Path = field(default_factory=lambda: Path("outputs"))

    # ── Dataset ──────────────────────────────────────────────────────────────
    image_size: Tuple[int, int] = (512, 512)
    num_train_samples: int = 1000
    num_test_samples: int = 67
    val_split: float = 0.10
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    # Grayscale mask values → class indices
    mask_value_to_class: Dict[int, int] = field(
        default_factory=lambda: {72: 0, 128: 1, 220: 2, 255: 3}
    )

    num_classes: int = 4
    class_names: Tuple[str, ...] = ("Roof", "Window", "Wall", "Other")
    class_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 128, 128),
    )

    # WWR-relevant class indices
    window_class: int = 1
    wall_class: int = 2

    # ── tf.data ──────────────────────────────────────────────────────────────
    batch_size: int = 8
    shuffle_buffer: int = 1024
    prefetch_buffer: Optional[int] = None  # None → tf.data.AUTOTUNE in dataset.py

    # ── Augmentation ─────────────────────────────────────────────────────────
    horizontal_flip: bool = True
    brightness_delta: float = 0.15
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    saturation_range: Tuple[float, float] = (0.85, 1.15)
    gaussian_noise_std: float = 0.02
    augment_probability: float = 0.5

    # ── Model ────────────────────────────────────────────────────────────────
    encoder_name: str = "efficientnetv2-s"
    encoder_weights: str = "imagenet"
    encoder_trainable: bool = True
    dropout_rate: float = 0.30
    l2_weight: float = 1e-4
    aspp_rates: Tuple[int, ...] = (6, 12, 18)
    aspp_filters: int = 256
    decoder_filters: Tuple[int, ...] = (256, 128, 64, 32)

    # ── Loss weights ─────────────────────────────────────────────────────────
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    boundary_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # ── Optimizer ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    total_epochs: int = 100
    min_learning_rate: float = 1e-7

    # ── Training ─────────────────────────────────────────────────────────────
    mixed_precision: bool = True
    xla_jit: bool = True
    early_stopping_patience: int = 15
    checkpoint_monitor: str = "val_mean_iou"
    checkpoint_mode: str = "max"
    enable_training_backup: bool = False  # BackupAndRestore uses extra disk space

    # Per-epoch visualization (Input | GT | Prediction)
    enable_epoch_visualization: bool = True
    viz_num_samples: int = 2
    viz_every_n_epochs: int = 1
    viz_show_in_notebook: bool = True

    # OneDrive (Colab via rclone) — 1 TB+ storage for model outputs
    use_onedrive: bool = False
    onedrive_mount_point: Path = field(default_factory=lambda: Path("/content/onedrive"))
    onedrive_project_dir: Path = field(
        default_factory=lambda: Path("/content/onedrive/WWR_Seg_Model")
    )
    rclone_config_path: Path = field(default_factory=lambda: Path("/content/rclone.conf"))
    rclone_remote_name: str = "onedrive"

    # ── Cross-validation ─────────────────────────────────────────────────────
    enable_cross_validation: bool = False
    cv_folds: int = 5

    # ── Inference ────────────────────────────────────────────────────────────
    best_model_filename: str = "best_model.keras"

    def __post_init__(self) -> None:
        """Resolve paths and create output directories."""
        self._resolve_paths()
        self._create_directories()

    def _resolve_paths(self) -> None:
        """Convert string paths and optionally remap to Google Drive / Colab layout."""
        path_fields = (
            "project_root",
            "data_root",
            "train_images_dir",
            "train_masks_dir",
            "test_images_dir",
            "test_masks_dir",
            "output_dir",
            "checkpoint_dir",
            "log_dir",
            "tensorboard_dir",
            "backup_dir",
            "predictions_dir",
            "wwr_output_dir",
            "drive_mount_point",
            "drive_project_dir",
            "drive_data_zip",
            "local_data_root",
            "models_dir",
            "results_dir",
            "onedrive_mount_point",
            "onedrive_project_dir",
            "rclone_config_path",
        )
        for name in path_fields:
            value = getattr(self, name)
            if not isinstance(value, Path):
                setattr(self, name, Path(value))

        if self.use_google_drive or self.use_colab:
            # Outputs → OneDrive (1 TB) if enabled, else Google Drive
            if self.use_onedrive:
                base = self.onedrive_project_dir
            else:
                base = self.drive_project_dir

            if self.use_local_data_cache:
                self.data_root = self.local_data_root
            else:
                self.data_root = base / "data"

            self.train_images_dir = self.data_root / "train" / "images"
            self.train_masks_dir = self.data_root / "train" / "masks"
            self.test_images_dir = self.data_root / "test" / "images"
            self.test_masks_dir = self.data_root / "test" / "masks"

            self.checkpoint_dir = base / "checkpoints"
            self.log_dir = base / "logs"
            self.models_dir = base / "models"
            self.results_dir = base / "results"
            self.output_dir = base / "results"
            self.predictions_dir = base / "results" / "predictions"
            self.wwr_output_dir = base / "results" / "wwr"
            self.tensorboard_dir = base / "results" / "tensorboard"
            self.backup_dir = base / "results" / "backup"

    def _create_directories(self) -> None:
        """Ensure all output directories exist."""
        for directory in (
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.tensorboard_dir,
            self.backup_dir,
            self.predictions_dir,
            self.wwr_output_dir,
            self.models_dir,
            self.results_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def best_model_path(self) -> Path:
        """Full path to the best saved model checkpoint."""
        if self.use_google_drive or self.use_colab:
            return self.models_dir / self.best_model_filename
        return self.checkpoint_dir / self.best_model_filename

    @classmethod
    def for_colab(cls, **overrides: object) -> "Config":
        """Preset for Google Colab: local data, Drive outputs."""
        defaults: dict = {
            "use_colab": True,
            "use_google_drive": True,
            "use_local_data_cache": True,
            "drive_project_dir": Path("/content/drive/MyDrive/WWR_Seg_Model"),
            "drive_data_zip": Path("/content/drive/MyDrive/WWR_Seg_Model/data.zip"),
            "local_data_root": Path("/content/data"),
            "batch_size": 8,
            "mixed_precision": True,
            "xla_jit": True,
            "seed": 42,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @property
    def mask_lut(self) -> List[int]:
        """256-entry lookup table mapping grayscale mask values to class indices."""
        lut = [255] * 256  # 255 = ignore / unknown
        for gray_value, class_idx in self.mask_value_to_class.items():
            lut[gray_value] = class_idx
        return lut

