# Semantic Segmentation with U-Net — Car, Cat, Bird

A semantic segmentation pipeline built with PyTorch that trains a U-Net model to segment **Cars**, **Cats**, and **Birds** in images. Data is sourced from the [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) dataset using [FiftyOne](https://voxel51.com/fiftyone/).

---

## Pipeline Overview

The notebook is organised into the following stages:

### 1. Data Download (`Cell 1`)
Downloads segmentation samples for Car, Cat, and Bird from Open Images v7 via FiftyOne Zoo. Each class is downloaded separately and merged into a single persistent FiftyOne dataset per split.

| Split      | Samples per class | Total (approx.) |
|------------|:-----------------:|:---------------:|
| Train      | 2000              | ~6000           |
| Validation | 300               | ~900            |
| Test       | 200               | ~600            |

### 1b. Background / Noise Download (`Cell 1B`)
Downloads images containing **none** of the three target classes to act as hard negatives. Images are filtered to discard any sample that accidentally contains a Car, Cat, or Bird annotation (e.g. a "Chair" image that also shows a Cat in the background).

| Split      | Noise samples |
|------------|:-------------:|
| Train      | 400           |
| Validation | 50            |
| Test        | 50            |

Noise classes used: Airplane, Table, Flower, Teapot, Laptop, Person, Book, Bottle, Apple, Clothing.

### 2. Colab / Local Setup
Two alternative setup paths are provided:

- **Colab**: mounts Google Drive, copies and unzips a pre-built `open_images.zip` archive from Drive, then loads datasets via `fo.Dataset.from_dir()`.
- **Local (`Cell 2`)**: loads already-downloaded FiftyOne persistent datasets (`oi_train`, `oi_validation`, `oi_test`) directly from disk. Raises an error if any required dataset is missing.

### 3. Mask Visualisation (`Cell 3`)
Displays one sample image per class with its ground-truth segmentation mask overlaid to verify the data loaded correctly. The `build_mask()` helper converts FiftyOne relative bounding-box patches to full-resolution pixel label maps on the fly.

### 4. Pipeline Setup (`Cell 4`)
Defines all global constants, random seeds, and helper functions:

- `CLASS_NAMES`, `CLASS_TO_ID`, `NUM_CLASSES` — class configuration
- `IMG_SIZE=256`, `BATCH_SIZE=8`, `MAX_EPOCHS=60`, `PATIENCE=10`
- `detect_label_field()` — auto-detects the FiftyOne field that holds segmentation labels
- `build_label_map_from_sample()` — converts FiftyOne detection masks to pixel-level label maps
- `collect_samples_with_target_classes()` — separates samples into *kept* (contains at least one target class) and *noise* (no target class) lists
- `verify_class_distribution()` / `check_pixel_distribution()` — reports per-class pixel counts and percentages with an ASCII bar chart

### 5. Noise Injection (`Cell 5a`) and Dataset Classes (`Cell 5b`)
- Noise samples (all-zero masks) are merged back into `train_samples`, `val_samples`, and `test_samples`.
- **`FiftyOneSegDataset`** — production PyTorch Dataset used for training:
  - Resizes images to 256×256 with bilinear interpolation; masks with nearest-neighbour
  - Normalises image pixels to `[0, 1]` (divide by 255)
  - Returns `float32` image tensor `[C, H, W]` and `int64` mask tensor `[H, W]`
- **`create_weighted_sampler()`** — builds a `WeightedRandomSampler` based on per-image foreground content:

  | Image content         | Weight |
  |-----------------------|:------:|
  | No foreground         | 0.2    |
  | Bird + Cat            | 8.0    |
  | Bird only             | 4.0    |
  | Car only              | 2.0    |
  | Cat only (no Car)     | 1.5    |
  | Mixed (other combos)  | 1.5    |

### 6. Data Augmentation (`Cell 6`)
Wraps the training dataset with online augmentation. Geometric transforms are applied identically to both image and mask; photometric transforms are image-only.

- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.3)
- Random 90° rotation (k ∈ {0,1,2,3}) — exact multiples avoid interpolation artefacts in the mask
- Saturation jitter ±30%, hue jitter ±0.1 (image only)
- Brightness / contrast jitter ±15% (image only)

### 7. U-Net Model (`Cell 7`)
A standard encoder–decoder U-Net with skip connections:

- **Encoder**: 4 downsampling stages with `DoubleConv` blocks (32→64→128→256 channels), each followed by MaxPool
- **Bottleneck**: 512-channel `DoubleConv`
- **Decoder**: 4 upsampling stages using transposed convolutions + skip concatenation; off-by-one size mismatches corrected with bilinear interpolation
- **Output**: 1×1 conv producing logits for 4 classes (background, Car, Cat, Bird)

Total trainable parameters: ~7.76 M (~29.6 MB).

### 8. Loss Function & Metrics (`Cell 8`)
Uses a balanced combination of Focal Loss and Dice Loss to address class imbalance:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Focal}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

- **Focal Loss** — focusing parameter γ=3; penalises hard, misclassified pixels more heavily
- **Dice Loss** — directly optimises pixel-level overlap between predicted and ground-truth masks
- **Manual class weights** applied to Focal Loss:

  | Class      | Weight |
  |------------|:------:|
  | Background | 0.05   |
  | Car        | 2.5    |
  | Cat        | 1.5    |
  | Bird       | 3.0    |

- **Optimiser**: AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- **Scheduler**: `CosineAnnealingLR` (T_max=MAX_EPOCHS, η_min=1e-7)

### 9. Training Loop (`Cell 9`)
Trains for up to 60 epochs with checkpoint resumption support:

- **Checkpoint resume**: if `latest_checkpoint.pt` exists in `CHECKPOINT_DIR`, training resumes from the last saved epoch (restores model, optimiser, scheduler, history, and early-stopping counter)
- **Best model**: saved to `best_model.pt` whenever `val_fg_f1` improves by more than 1e-4
- **Latest checkpoint**: overwritten every epoch to ensure resumability
- **Early stopping**: stops after 10 epochs without improvement in val foreground macro-F1
- **Metric tracked**: foreground macro-F1 across Car, Cat, Bird (background excluded)
- Plots loss, pixel accuracy, and foreground macro-F1 curves after training

### 10. Checkpoint Save/Load (`Cell 10`)
Saves the best model to `checkpoints/unet_car_cat_bird_best_final2.pt` (Google Drive path in Colab) including:

- `model_state_dict` — trained weights
- `num_classes`, `img_size`, `class_names`, `class_to_id` — metadata for reconstruction

### 11. Load from Previous Session (`Cell 11 — "After closing"`)
Standalone cell to restore a saved model without re-running training. Restores all metadata from the checkpoint and sets the model to `eval()` mode. Only requires the `UNet` class to be defined.

### 12. Evaluation (`Cell 12`)
Runs inference on the test set and computes:

- **Mean IoU per class** (background, Car, Cat, Bird) and overall mIoU
- **Residual map analysis** — visualises the 3 worst-performing test images (highest pixel error rate)

### 13. Per-Class Metrics (`Cell 13`)
Computes pixel-level, one-vs-rest metrics for each foreground class across the full test set:

- **Accuracy**, **Precision**, **Recall** (recovery rate), **F1**
- Macro-averaged totals printed across all three classes

### 14. Test Set Prediction Visualisation (`Cell 14`)
Scans the test set and picks the most representative image per class — the one where that class covers the most ground-truth pixels. Displays 4 panels per sample: Original image | Ground truth mask | Predicted mask | Overlay.

### 15. External Image Inference (`Cell 15`)
Runs the trained model on any user-provided image file:

```python
your_photo_path = r"C:\path\to\your\image.jpg"
predict_external_image(model, your_photo_path, device)
```

The prediction mask is upsampled back to the original image resolution (nearest-neighbour) before overlay, ensuring correct spatial alignment.

### 16. Pixel-Level Coordinate Query
An additional utility cell runs the model and reports the predicted class and full softmax confidence scores for a single pixel coordinate (default: x=123, y=123). Useful for debugging or interactive exploration.

---

## Requirements

```
torch
torchvision
fiftyone
Pillow
numpy
matplotlib
torchsummary
```

Install with:

```bash
pip install torch torchvision fiftyone Pillow numpy matplotlib torchsummary
```

---

## Class Colour Map

| Class      | ID | Colour        |
|------------|:--:|---------------|
| Background | 0  | Black         |
| Car        | 1  | Red           |
| Cat        | 2  | Deep Sky Blue |
| Bird       | 3  | Lime          |

---

## Notes

- In Colab, checkpoints are saved to `Google Drive/MyDrive/segmentation_checkpoints/` and persist across runtime disconnects
- Locally, images are downloaded to the FiftyOne default directory (configurable in Cell 1)
- The model input is always resized to **256×256** — predictions are upsampled back to original resolution for overlay visualisation
- Training supports **resumption**: re-running Cell 9 will pick up from the last saved epoch if `latest_checkpoint.pt` exists
