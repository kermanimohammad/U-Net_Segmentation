# U-Net Segmentation for WWR Estimation

## Overview
The **U-Net Segmentation module** is a deep learning-based solution for automatically detecting and segmenting **windows, walls, roofs, and other parts** of buildings from facade images. It utilizes a **U-Net model with MobileNetV2 as the feature extractor**, trained to perform **multi-class semantic segmentation**.

## Features
- **Deep Learning-Based Segmentation:** Uses a **U-Net model** to classify pixels as windows, walls, roofs, or other.
- **Preprocessing and Augmentation:** Applies **image normalization, resizing, random flipping, brightness & contrast adjustment, and rotation**.
- **Fine-Tuned MobileNetV2 Backbone:** Uses MobileNetV2 layers to extract image features before segmentation.
- **Custom Loss Function:** Combines **Categorical Crossentropy and Dice loss** for improved segmentation accuracy.
- **Visualization Tools:** Uses `matplotlib` to visualize predictions.

## Installation

### Prerequisites
Ensure you have the following installed:
- **Python** (>= 3.8)
- **pip** (Python package manager)
- **TensorFlow** (>= 2.12)
- **OpenCV**
- **Matplotlib**
- **NumPy**
- **scikit-image**
- **scikit-learn**

### Steps
1. Clone the repository:
   ```sh
   git clone https://your-university-gitea-link/u-net-utilization.git
   cd u-net-utilization
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. **Place Images in Folders:**
   - Raw images: `./data/png_images/`
   - Corresponding segmentation masks: `./data/png_masks/`
2. **Ensure masks have proper labels:**
   - Roofs: `0`
   - Windows: `1`
   - Walls: `2`
   - Other: `3`

## Training the U-Net Model
Run the training script:
```sh
python u_net_segmentation.py
```
- The model will be trained for **30 epochs** with **batch size 8**.
- The trained model will be saved in `./models/wwr.keras`.

## Model Architecture
- **Encoder:** Uses **MobileNetV2** as the backbone for feature extraction.
- **Decoder:** Uses transposed convolutions with skip connections.
- **Final Layer:** Applies a **softmax activation** to produce segmentation probabilities.

## Running Inference
After training, you can test the model on new images:
```sh
python u_net_segmentation.py --test
```
This will:
- Load test images from `./data/test_images/final_images/`
- Predict masks using the trained model.
- Save the predicted segmentation maps.

## Folder Structure
```
u-net-utilization/
├── data/
│   ├── png_images/          # Training images
│   ├── png_masks/           # Corresponding segmentation masks
│   ├── test_images/         # Test images for inference
├── models/
│   ├── wwr.keras            # Trained U-Net model
├── u_net_segmentation.py    # Training and inference script
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
```

## Development
- **Train the model:**
  ```sh
  python u_net_segmentation.py
  ```
- **Test the model on sample images:**
  ```sh
  python u_net_segmentation.py --test
  ```

## Dependencies
- TensorFlow (>= 2.12)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- scikit-image

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push the branch: `git push origin feature-branch`
5. Open a Pull Request.

## License
This project is licensed under [MIT License](LICENSE).

## Acknowledgments
- **Supervisor:** Professor Ursula Eicker
- **Research Group:** Next Generation Cities Institute
