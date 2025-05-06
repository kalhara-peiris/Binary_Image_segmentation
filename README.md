# Shoreline Segmentation Model

A deep learning project for automated shoreline extraction from satellite imagery using a DeepLabV3+ architecture implemented from scratch.

## Overview

This repository contains a PyTorch implementation of a semantic segmentation model designed to identify and extract shorelines from satellite imagery. The model uses a DeepLabV3+ architecture built entirely from scratch with memory optimizations and is trained to create binary masks that differentiate between water and land areas.

## Features

- **Custom DeepLabV3+ Architecture**: Implemented from scratch and adapted for memory efficiency with reduced stride and GroupNorm
- **Mixed Precision Training**: Uses PyTorch AMP for memory-efficient training
- **Memory Optimizations**: Includes gradient clipping, memory clearing, and efficient data loading
- **Advanced Loss Functions**: Combines Focal Loss and Dice Loss for better boundary detection
- **Data Augmentation**: Comprehensive image augmentation pipeline using Albumentations
- **Early Stopping**: Prevents overfitting with patience-based early stopping

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)
- Google Colab (for the current implementation)
- Google Drive (for storage)

## Dependencies

```
torch
torchvision
numpy
opencv-python (cv2)
albumentations
scikit-learn
tqdm
PIL
```

## Directory Structure

The code expects the following directory structure in your Google Drive:

```
/content/drive/MyDrive/
├── dataset/
│   ├── training_satellite/      # Training satellite images
│   ├── training_mask/           # Training mask images
│   ├── validation_satellite/    # Validation satellite images
│   ├── validation_mask/         # Validation mask images
│   ├── testing_satellite/       # Test satellite images
│   └── testing_mask/            # Test mask images
└── shoreline_model_bs/          # Model checkpoints and training history
```

## Model Architecture

The model architecture is implemented from scratch and consists of:

1. **Backbone**: Custom ResNet-like network with GroupNorm
2. **ASPP Module**: Custom Atrous Spatial Pyramid Pooling with dilated convolutions
3. **Decoder**: Feature fusion decoder that combines high-level and low-level features
4. **Auxiliary Head**: Additional supervision during training

All components are built from the ground up without using pre-trained models or existing implementations.

## Training

The model is trained with:

- Batch size: 8
- Image size: 1024×1024 (center-cropped)
- Learning rate: 0.0003
- Optimizer: AdamW with weight decay
- Scheduler: Cosine Annealing
- Loss: Combined Focal Loss and Dice Loss
- Epochs: 200 (with early stopping)

## Usage

1. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Ensure your dataset is organized in the expected directory structure

3. Run the main script:
   ```python
   python main.py
   ```

## Outputs

The model saves:

- Best model weights (`best_shoreline_deeplabv3.pt`)
- Final model weights (`final_shoreline_deeplabv3.pt`)
- Training history (`training_history_deeplabv3.pkl` and `final_training_history_deeplabv3.pkl`)

## Custom Components

### Loss Function

The repository implements a custom loss function that combines Focal Loss and Dice Loss:

- **Focal Loss**: Addresses class imbalance by focusing on hard examples
- **Dice Loss**: Improves boundary detection by optimizing for pixel-wise overlap

### Dataset Class

The `ShorelineDataset` class handles loading and processing of image-mask pairs, with support for on-the-fly augmentations.

### Early Stopping

The `EarlyStopping` class monitors validation loss and stops training when performance plateaus to prevent overfitting.

## Acknowledgments

This implementation uses techniques from state-of-the-art segmentation literature, including:

- DeepLabV3+ architecture (implemented entirely from scratch)
- Memory-efficient training practices
- Advanced loss functions for boundary detection

Unlike many repositories that use pre-trained backbones or existing implementations, this project builds the entire DeepLabV3+ architecture from the ground up, allowing for complete customization and optimization for the shoreline segmentation task.

