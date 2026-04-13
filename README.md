# Semantic Segmentation with U-Net — Car, Cat, Bird

A semantic segmentation pipeline built with PyTorch that trains a U-Net model to segment **Cars**, **Cats**, and **Birds** in images. Data is sourced from the [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) dataset using [FiftyOne](https://voxel51.com/fiftyone/).

---


## Pipeline Overview

The notebook is organized into the following stages:

### 1. Data Download (`Cell 1`)
Downloads segmentation samples for Car, Cat, and Bird from Open Images v7 via FiftyOne Zoo, and stores them as persistent FiftyOne datasets on disk.

| Split      | Samples per class | Total (approx.) |
|------------|:-----------------:|:---------------:|
| Train      | 400               | ~1200           |
| Validation | 50                | ~150            |
| Test       | 100               | ~300            |

### 2. Dataset Loading (`Cell 2`)
Loads the previously downloaded FiftyOne datasets back into memory. Run this instead of Cell 1 if data is already on disk.

### 3. Mask Visualization (`Cell 3`)
Displays one sample image per class with its ground-truth segmentation mask overlaid, to verify the data loaded correctly.

### 4. Pipeline Setup (`Cell 4`)
Defines all global constants, random seeds, and helper functions:
- `CLASS_NAMES`, `CLASS_TO_ID`, `NUM_CLASSES` — class configuration
- `IMG_SIZE=256`, `BATCH_SIZE=8`, `MAX_EPOCHS=40`, `PATIENCE=7`
- `build_label_map_from_sample()` — converts FiftyOne detection masks to pixel-level label maps
- `collect_samples_with_target_classes()` — filters samples that contain at least one target class
- `verify_class_distribution()` / `create_stratified_split()` — ensures val/test splits contain all classes

### 5. Dataset Classes (`Cells 5–6`)
- **`SegmentationDataset`** — basic PyTorch Dataset wrapping FiftyOne samples
- **`FiftyOneSegDataset`** — production dataset used for training, with proper resizing (256×256) and float normalization `[0, 1]`
- **`create_weighted_sampler()`** — creates a `WeightedRandomSampler` that oversamples minority classes (Cat, Bird) during training

### 6. Data Augmentation (`Cell 7`)
Wraps the training dataset with online augmentation:
- Random horizontal/vertical flips
- Random 90° rotations (identical transforms applied to image and mask)
- Mild brightness/contrast jitter (image only, mask unchanged)

### 7. U-Net Model (`Cell 8`)
A standard encoder–decoder U-Net with skip connections:
- **Encoder**: 4 downsampling stages with `DoubleConv` blocks (32→64→128→256 channels)
- **Bottleneck**: 512-channel `DoubleConv`
- **Decoder**: 4 upsampling stages using transposed convolutions + skip concatenation
- **Output**: 1×1 conv producing logits for 4 classes (background, Car, Cat, Bird)

### 8. Loss Function & Metrics (`Cell 9`)
Uses a combined loss to address class imbalance:

$$\mathcal{L} = 0.3 \cdot \mathcal{L}_{\text{Focal}} + 0.7 \cdot \mathcal{L}_{\text{Dice}}$$

- **Focal Loss** — penalises hard, misclassified pixels more heavily
- **Dice Loss** — directly optimises overlap between predicted and ground-truth masks
- **Class weights** — estimated from training data frequency; minority classes (Cat, Bird) boosted ×3

### 9. Training Loop (`Cell 10`)
Trains for up to 40 epochs with:
- **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-5`)
- **Scheduler**: `ReduceLROnPlateau` — halves LR if val loss stalls for 2 epochs
- **Early stopping**: stops after 7 epochs without improvement in val foreground macro-F1
- Plots loss, pixel accuracy, and foreground macro-F1 curves after training

### 10. Checkpoint Save/Load (`Cell 11`)
Saves the best model to `checkpoints/unet_person_cat_bird_best.pt` including:
- `model_state_dict` — trained weights
- `num_classes`, `img_size`, `class_names`, `class_to_id` — metadata for reconstruction

### 11. Load from Previous Session (`Cell 12 — "After closing"`)
Standalone cell to restore a saved model without re-running training. Only requires the `UNet` class to be defined.

### 12. Evaluation (`Cell 13`)
Runs inference on the test set and computes:
- **Mean IoU per class** (background, Car, Cat, Bird) and overall mIoU
- **Residual map analysis** — visualises the 3 worst-performing test images (highest pixel error rate)

### 13. Per-Class Metrics (`Cell 14`)
Computes pixel-level, one-vs-rest metrics for each foreground class:
- **Accuracy**, **Precision**, **Recall** (recovery), **F1**
- Prints macro-averaged totals across all three classes

### 14. Test Set Prediction Visualization (`Cell 15`)
Displays one representative test image per class (Car, Cat, Bird) with:
- Original image | Ground truth mask | Predicted mask | Overlay

### 15. External Image Inference (`Cell 16`)
Runs the trained model on any user-provided image file:
```python
your_photo_path = r"C:\path\to\your\image.jpg"
predict_external_image(model, your_photo_path, device)
```
The mask is upsampled back to the original image resolution before overlay.

---

## Quick Start

### Full Training Run
Run cells in order: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11**

### Inference on Saved Checkpoint (No Re-Training)
1. Run the **`UNet` model definition cell** (cell 8) to define the architecture
2. Run the **"After closing" loader cell** to restore weights and metadata
3. Run the **external image inference cell** with your photo path

---

## Requirements

```
torch
torchvision
fiftyone
Pillow
numpy
matplotlib
```

Install with:
```bash
pip install torch torchvision fiftyone Pillow numpy matplotlib
```

---

## Class Colour Map

| Class      | ID | Colour      |
|------------|:--:|-------------|
| Background | 0  | Black       |
| Car        | 1  | Red         |
| Cat        | 2  | Deep Sky Blue |
| Bird       | 3  | Lime        |

---

## Notes
- Images are downloaded to `D:/open_images` by default (configurable in Cell 1)
- Training uses a `WeightedRandomSampler` to oversample Cat and Bird images, which are rarer than Car in Open Images
- The model input is always resized to **256×256** — predictions are upsampled back to original resolution for overlay visualization
