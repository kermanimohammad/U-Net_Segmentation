# WWR Segmentation

Production-quality semantic segmentation pipeline for extracting **Roof**, **Window**, **Wall**, and **Other** building classes from Google 3D Mesh facade textures, with **Window-to-Wall Ratio (WWR)** estimation.

Designed for research publication (Intelligent Computing / Science Partner Journal) with reproducibility, modularity, and segmentation accuracy as top priorities.

## Architecture

- **Encoder:** EfficientNetV2-S (ImageNet pretrained)
- **Bridge:** ASPP (Atrous Spatial Pyramid Pooling)
- **Decoder:** Residual blocks with skip fusion, BatchNorm, Swish, SpatialDropout2D
- **Loss:** 0.5 Dice + 0.3 Focal + 0.2 Boundary
- **Optimizer:** AdamW with warmup-cosine LR, gradient clipping, mixed precision, XLA

## Installation

```bash
git clone https://github.com/your-org/WWR_Segmentation.git
cd WWR_Segmentation
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r WWR_Segmentation/requirements.txt
```

### Google Colab (A100)

```python
!pip install -r WWR_Segmentation/requirements.txt
from WWR_Segmentation.config import Config
config = Config(use_google_drive=True)
```

## Dataset Layout

```
data/
├── train/
│   ├── images/     # 1000 RGB images
│   └── masks/      # 1000 grayscale masks
└── test/
    ├── images/     # 67 RGB images (NEVER used in training)
    └── masks/      # 67 grayscale masks
```

### Mask Grayscale Values

| Value | Class  | Index |
|-------|--------|-------|
| 72    | Roof   | 0     |
| 128   | Window | 1     |
| 220   | Wall   | 2     |
| 255   | Other  | 3     |

## Usage

### Training

```python
from WWR_Segmentation.config import Config
from WWR_Segmentation.trainer import train

config = Config(
    data_root="data",
    train_images_dir="data/train/images",
    train_masks_dir="data/train/masks",
    batch_size=8,
    total_epochs=100,
)
model, history = train(config)
```

### Evaluation

```python
from WWR_Segmentation.evaluate import run_evaluation

results = run_evaluation(config)
```

### Inference

```python
from WWR_Segmentation.inference import run_inference

run_inference(config, input_dir="data/test/images")
```

### WWR Calculation

```python
from WWR_Segmentation.wwr import run_wwr_analysis

df = run_wwr_analysis(config)
```

### 5-Fold Cross-Validation

```python
from WWR_Segmentation.cross_validation import run_cross_validation

config.enable_cross_validation = True
summary = run_cross_validation(config)
```

## Project Structure

```
WWR_Segmentation/
├── config.py              # All hyperparameters and paths
├── dataset.py             # tf.data pipeline
├── augmentation.py        # Realistic augmentations
├── model.py               # EfficientNetV2-S + ASPP + Decoder
├── aspp.py                # ASPP bridge module
├── decoder.py             # Residual decoder
├── losses.py              # Dice, Focal, Boundary losses
├── metrics.py             # IoU, Dice, F1, confusion matrix
├── optimizer.py           # AdamW + warmup-cosine LR
├── callbacks.py           # Checkpoint, TensorBoard, etc.
├── trainer.py             # Training orchestration
├── evaluate.py            # Validation & test evaluation
├── cross_validation.py    # Optional 5-fold CV
├── wwr.py                 # Window-to-Wall Ratio module
├── inference.py           # Single & batch inference
├── utils.py               # Seeding, logging, helpers
├── requirements.txt
└── README.md
WWR_Seg_Model.ipynb        # Clean demonstration notebook
```

## Configuration

All parameters are configurable through `Config`:

```python
from WWR_Segmentation.config import Config

config = Config(
    seed=42,
    image_size=(512, 512),
    batch_size=8,
    learning_rate=1e-4,
    mixed_precision=True,
    use_google_drive=True,  # Colab + Google Drive
)
```

## Outputs

All artifacts are saved under `outputs/` (or Google Drive):

- `checkpoints/best_model.keras` — best model by validation mean IoU
- `logs/` — CSV training history
- `tensorboard/` — TensorBoard logs
- `predictions/` — confusion matrices, galleries, failure cases
- `wwr/` — WWR CSV, Excel, summary, distribution plot

## License

MIT License — see [LICENSE](../LICENSE).

## Acknowledgments

- **Supervisor:** Professor Ursula Eicker
- **Research Group:** Next Generation Cities Institute, Concordia University
