# Shoreline Segmentation with DeepLabV3+

This repository contains a PyTorch implementation of DeepLabV3+ for shoreline segmentation in satellite/aerial imagery. The model is designed to accurately identify and segment shorelines in coastal imagery, which can be useful for coastal monitoring, erosion analysis, and environmental studies.

## Project Overview

The project implements a custom DeepLabV3+ architecture with ResNet-like backbone for semantic segmentation of shorelines. Key features include:

- Custom DeepLabV3+ implementation with modified stride and ASPP (Atrous Spatial Pyramid Pooling)
- Combined Focal and Dice loss for handling class imbalance
- Data augmentation pipeline using Albumentations
- Training pipeline with early stopping and learning rate scheduling
- Google Colab integration for easy model training and storage

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- scikit-learn
- tqdm
- Google Colab (for Drive integration)

## Installation

```bash
pip install torch torchvision numpy matplotlib opencv-python albumentations scikit-learn tqdm
```

## Dataset Structure

The code expects the following directory structure:

```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

- Images should be RGB images
- Masks should be binary grayscale images where white (255) represents the shoreline and black (0) represents the background

## Model Architecture

The implemented DeepLabV3+ architecture includes:

- ResNet-like backbone with modified strides
- ASPP module with dilation rates of 6, 12, and 18
- Low-level feature integration from early layers
- Auxiliary decoder for deep supervision during training

## Training

To train the model:

```python
# Define data directories
train_image_dir = '/path/to/train/images'
train_mask_dir = '/path/to/train/masks'
val_image_dir = '/path/to/val/images'
val_mask_dir = '/path/to/val/masks'
test_image_dir = '/path/to/test/images'
test_mask_dir = '/path/to/test/masks'

# Prepare dataloaders
train_loader, val_loader, test_loader = prepare_dataloaders(
    train_image_dir, train_mask_dir,
    val_image_dir, val_mask_dir,
    test_image_dir, test_mask_dir,
    batch_size=6, image_size=540
)

# Initialize model
model = DeepLabV3Plus(num_classes=1).to(device)

# Train model
model, training_history = train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=200,
    learning_rate=0.0003
)
```

## Hyperparameters

The default hyperparameters are:
- Learning rate: 0.0003
- Batch size: 6
- Number of epochs: 200
- Image size: 540x540
- Optimizer: AdamW with weight decay 1e-4
- Scheduler: CosineAnnealingLR
- Loss function: Combined Focal Loss (alpha=0.003, gamma=5.0) and Dice Loss (weight=0.5)

## Loss Function

The model uses a custom loss function that combines Focal Loss and Dice Loss:

- **Focal Loss**: Helps focus on hard-to-classify examples
- **Dice Loss**: Addresses class imbalance by directly optimizing the overlap between predictions and ground truth

## Data Augmentation

The training pipeline includes extensive data augmentation:
- Horizontal and vertical flips
- Random rotations
- Elastic transformations
- Grid and optical distortions
- Gaussian, median, and motion blur
- Brightness and contrast adjustments
- Hue and saturation changes

## Early Stopping

The training process includes early stopping with:
- Patience: 25 epochs
- Minimum delta: 1e-4

## Model Saving

Models are saved in two ways:
- Best model based on validation loss
- Final model after training

Training history is saved periodically and at the end of training.

## Google Drive Integration

The code automatically mounts Google Drive and creates a directory for saving models:

```python
MODEL_SAVE_PATH = '/content/drive/MyDrive/shoreline_models16sss'
```

