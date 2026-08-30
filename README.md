# VisionInspectAI
Use MVTec-AD categories to detect whether an image is normal or defective, and show the defect location using a heatmap. Currently trained/evaluated end-to-end on `screw`, `bottle`, `hazelnut`, `carpet`, `leather`, `wood`, `grid`, `tile`, and `transistor`; the Streamlit demo auto-detects which one was uploaded.

## Getting Started (New Clone Setup)

These steps take a fresh clone from zero to a running Streamlit demo. **Note:** the raw MVTec-AD dataset, trained model checkpoints (`models/checkpoints/`) and generated outputs (`outputs/`) are all excluded via `.gitignore` (too large for git) — only source code and the small `data/manifests/*.csv` files are tracked, so you need to download the dataset and (re)train the models yourself after cloning.

### 1. Clone and set up a Python environment

```bash
git clone <this-repo-url>
cd VisionInspectAI

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirement.txt
```

Requires Python 3.9+.

### 2. Download the MVTec-AD dataset

Download the dataset from the official [MVTec-AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) (free for research/non-commercial use — see `data/mvtec_anomaly_detection/license.txt`) and extract it so the folder layout looks like:

```
data/mvtec_anomaly_detection/
	<category>/
		train/good/...
		test/<defect_type or good>/...
		ground_truth/<defect_type>/...
```

At minimum, grab the categories this project is already configured for: `screw`, `bottle`, `hazelnut`, `carpet`, `leather`, `wood`, `grid`, `tile`, and `transistor`. (You only need the top-level dataset archive, or the individual per-category archives for these nine.)

### 3. Build manifests, train the models, and launch the demo

Repeat for each category (`screw`, `bottle`, `hazelnut`, `carpet`, `leather`, `wood`, `grid`, `tile`, `transistor`):

```bash
python -m src.data.create_manifest --category screw
python -m src.models.train_baseline --config config/screw_config.yaml
python -m src.models.run_anomaly_detection --config config/screw_config.yaml
```

Then train the category classifier (needed for the Streamlit demo's auto-detect feature) once all nine manifests exist:

```bash
python -m src.models.train_category_classifier --categories screw bottle hazelnut carpet leather wood grid tile transistor
```

Optionally, train each category's defect-type classifier (needed for the Streamlit demo to show *what kind* of defect was found, not just Normal/Defective):

```bash
python -m src.models.train_defect_classifier --config config/screw_config.yaml
```

Finally, launch the demo:

```bash
streamlit run app/streamlit_app.py
```

### 4. Verify everything works

```bash
python -m pytest tests/ -v
```

See [Quick Start](#quick-start) below for the condensed command list and [Pipeline Steps in Detail](#pipeline-steps-in-detail) for what each script/output actually is.

## Learning & Techniques Involved

This project combines multiple pattern recognition / machine learning aspects, applied to several MVTec-AD categories:

- **Supervised / unsupervised learning scenarios**
  - Supervised: [src/models/baseline_classifier.py](src/models/baseline_classifier.py) — a `resnet18`, `efficientnet_b0`, `convnext_tiny`, or `vit_b_16` transfer-learning classifier, or `simple_cnn` trained from scratch, fine-tuned on labeled good/defective images (trained via [src/models/train_baseline.py](src/models/train_baseline.py)).
  - Unsupervised: [src/models/anomaly_detector.py](src/models/anomaly_detector.py) — a PatchCore-style anomaly detector trained only on `train/good` images (no defect labels needed), evaluated on the full labeled test set via [src/models/run_anomaly_detection.py](src/models/run_anomaly_detection.py).

- **Machine learning / deep learning techniques**
  - ImageNet-pretrained deep backbones: ResNet18, EfficientNet-B0, and ConvNeXt-Tiny CNNs plus ViT-B/16, used for supervised fine-tuning comparisons; ResNet18/WideResNet50-2 are also used as frozen PatchCore feature extractors.
  - Classical ML techniques layered on top of deep features: greedy k-center coreset subsampling and nearest-neighbor distance scoring for anomaly detection ([src/models/anomaly_detector.py](src/models/anomaly_detector.py)), random/PCA dimensionality reduction for coreset selection, a PCA + Linear Discriminant Analysis classifier on frozen CNN embeddings as an alternative per-category defect-type classifier ([src/models/train_pca_lda_defect_classifier.py](src/models/train_pca_lda_defect_classifier.py), see [Classical ML alternative](#classical-ml-alternative-pca--lda-defect-type-classifier) below), plus standard evaluation metrics (accuracy/precision/recall/F1, ROC-AUC, IoU, Dice) in [src/evaluation/metrics.py](src/evaluation/metrics.py).

- **Hybrid machine learning / ensemble approach**
  - The pipeline combines a deep feature extractor (CNN) with a classical nearest-neighbor memory bank (PatchCore), i.e. deep representation learning + non-parametric matching, rather than a purely end-to-end network.
  - Two complementary models — the supervised classifier and the unsupervised anomaly detector — are trained and evaluated side by side on the same data, giving two independent good/defective signals that can be cross-checked.

- **Intelligent sensing / sense making techniques**
  - Image preprocessing and augmentation pipeline ([src/preprocessing/transform.py](src/preprocessing/transform.py)) turns raw camera/sensor images into model-ready tensors.
  - Pixel-level "sense making": anomaly heatmaps ([src/visualization/heatmap.py](src/visualization/heatmap.py)) localize the suspected defect region from raw pixel-level anomaly scores, compared against ground-truth defect masks (pixel AUROC, IoU, Dice).
  - Foreground/object segmentation ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py)) senses which pixels belong to the physical part vs. the background, so scoring and heatmaps focus on the object being inspected (where applicable — see the Generalization section below).
  - The [app/streamlit_app.py](app/streamlit_app.py) demo auto-detects which object type was uploaded ([src/models/train_category_classifier.py](src/models/train_category_classifier.py)) and turns it into an actionable decision: Normal/Defective prediction, anomaly score, severity (Low/Medium/High), and a heatmap overlay of the likely defect region — no manual category selection required.

## Project Structure

- `src/data/` — manifest generation ([create_manifest.py](src/data/create_manifest.py)) and PyTorch `Dataset` ([dataset.py](src/data/dataset.py))
- `src/preprocessing/` — image transforms and foreground/object segmentation
- `src/models/` — baseline classifiers, PatchCore anomaly detector, ensemble fusion, and training/evaluation scripts
- `src/evaluation/` — classification/pixel-level metrics and seeded benchmark aggregation
- `src/visualization/` — anomaly heatmap rendering
- `notebooks/` — dataset exploration
- `app/` — Streamlit inspection demo
- `config/` — per-category YAML configs (e.g. `screw_config.yaml`)
- `tests/` — unit tests for evaluation/preprocessing/visualization functions and baseline-training controls

## Quick Start

```bash
# 1. Build the manifest CSV for a category
python -m src.data.create_manifest --category screw

# 2. Train the supervised baseline classifier
python -m src.models.train_baseline --config config/screw_config.yaml

# Optional: obtain a more reliable mean ± standard deviation with 5-fold CV
python -m src.models.train_baseline --config config/screw_config.yaml --cross-validation --folds 5

# 3. Train and evaluate the unsupervised PatchCore anomaly detector
python -m src.models.run_anomaly_detection --config config/screw_config.yaml

# 4. (optional) Train the per-category defect-type classifier (what kind of defect is it?)
python -m src.models.train_defect_classifier --config config/screw_config.yaml

# Screw option: train on deployment-matched PatchCore defect crops
python -m src.models.train_defect_classifier --config config/screw_config.yaml \
  --defect-focused-crops --crop-source patchcore
**Algorithm:** the same `build_baseline_model` architecture as the category's `config/<category>_config.yaml` (ResNet18 by default), with an `N`-way head (`N` = number of defect types for that category), trained with cross-entropy + Adam for 10 epochs. Training seeds NumPy/PyTorch from `data.seed` so full-image and crop experiments are reproducible.

Add `--defect-focused-crops --crop-source patchcore` to crop each image around the region above the saved PatchCore pixel threshold before resize/augmentation. This uses the same predicted localization available in Streamlit, with 25% context padding and a minimum crop side of 25% of the shorter image dimension by default. `--crop-source ground-truth` is available as an oracle ablation, but its validation score is not a deployment estimate because uploaded images have no ground-truth masks.

# 5. Launch the interactive demo (auto-detects the category from the uploaded image)
streamlit run app/streamlit_app.py

# 6. Run the unit tests
python -m pytest tests/ -v
```

### Optional — RBF, GRNN, and SOM comparison

The project also includes embedding-based Radial Basis Function (RBF),
General Regression Neural Network (GRNN), and Self-Organizing Map (SOM)
models. All use fixed, ImageNet-pretrained ResNet18 512-dimensional
embeddings, then fit a shallow model on those features. This makes the
comparison fair without treating raw image pixels as a kernel input. The
train/validation split matches the corresponding CNN experiment for each
classification task.

Run both classifiers for the supervised good/defective baseline:

```bash
python -m src.models.train_kernel_classifiers \
  --task baseline --config config/screw_config.yaml
```

Run object category classification:

```bash
python -m src.models.train_kernel_classifiers \
  --task category --categories screw bottle hazelnut carpet leather grid tile wood
```

Run per-category defect-type classification:

```bash
python -m src.models.train_kernel_classifiers \
  --task defect --config config/screw_config.yaml
```

Use `--classifier rbf`, `--classifier grnn`, or `--classifier som` to run one
model. With `--classifier both`, all three models are trained. Each run writes
a separate pickle checkpoint, metrics JSON, and confusion-matrix figure under
`models/checkpoints/`, `outputs/metrics/`, and `outputs/figures/`, respectively.
SOM runs also write a U-Matrix image.

For unsupervised SOM anomaly exploration, train the map on `train/good` only
and score the labeled test set using quantization error:

```bash
python -m src.models.train_kernel_classifiers \
  --task anomaly --config config/screw_config.yaml
```

This produces image-level anomaly metrics and compares the SOM ROC-AUC with
the stored PatchCore ROC-AUC when that metric file exists. SOM anomaly
exploration does not produce pixel heatmaps; PatchCore remains the model used
for unsupervised defect localization.

#### Measured comparison results

These runs used the existing validation protocols and fixed ResNet18
embeddings. The screw baseline is directly comparable with the supervised
ResNet18 and Simple CNN results above; the category and defect results use the
eight-category manifests now tracked by the project. **Note:** `transistor` was
added in a separate branch after this comparison was run and is not yet
included in these RBF/GRNN/SOM numbers.

| Task / split | Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---:|---:|---:|
| Screw good/defective | ResNet18 | 1.000 | 1.000 | n/a |
| Screw good/defective | Simple CNN | 1.000 | 1.000 | n/a |
| Screw good/defective | RBF + ResNet18 embeddings | 0.833 | 0.764 | 0.824 |
| Screw good/defective | GRNN + ResNet18 embeddings | 0.708 | 0.681 | 0.660 |
| Eight-category object classification | RBF + ResNet18 embeddings | 1.000 | 1.000 | 1.000 OVR |
| Eight-category object classification | GRNN + ResNet18 embeddings | 0.977 | 0.976 | 0.989 OVR |
| Eight-category object classification | SOM + ResNet18 embeddings | 1.000 | 1.000 | 1.000 OVR |

Per-category defect-type validation results were:

| Category | RBF accuracy / macro F1 | GRNN accuracy / macro F1 | SOM accuracy / macro F1 |
|---|---:|---:|---:|
| Screw | 0.556 / 0.524 | 0.361 / 0.356 | 0.194 / 0.188 |
| Bottle | 0.842 / 0.842 | 0.579 / 0.463 | 0.789 / 0.775 |
| Hazelnut | 0.714 / 0.706 | 0.286 / 0.183 | 0.571 / 0.576 |
| Carpet | 0.704 / 0.698 | 0.333 / 0.283 | 0.481 / 0.471 |
| Leather | 0.786 / 0.762 | 0.286 / 0.267 | 0.714 / 0.706 |
| Grid | 0.333 / 0.311 | 0.167 / 0.197 | 0.278 / 0.209 |
| Tile | 0.885 / 0.875 | 0.462 / 0.420 | 0.731 / 0.738 |
| Wood | 0.889 / 0.863 | 0.444 / 0.392 | 0.556 / 0.496 |

SOM quantization-error anomaly ROC-AUC was: screw `0.514`, bottle `0.983`,
hazelnut `0.911`, carpet `0.795`, leather `0.921`, grid `0.476`, tile
`0.978`, and wood `0.885`. The matching stored PatchCore image ROC-AUC values
were screw `0.909`, bottle `1.000`, hazelnut `0.998`, carpet `0.971`, leather
`1.000`, grid `1.000`, tile `1.000`, and wood `1.000`.

**SOM interpretation by task:**

- **Good/defective baseline:** SOM is a weak primary classifier here. On
  screw it reached `0.708` accuracy but only `0.500` ROC-AUC, versus RBF's
  `0.833` and `0.824`. The map preserves embedding topology but does not
  directly learn the binary decision boundary.
- **Object category classification:** SOM is highly useful. Whole object
  categories are well separated in ResNet embedding space, so majority-label
  SOM nodes achieved `1.000` accuracy and OVR ROC-AUC, matching RBF and
  exceeding GRNN's `0.977` accuracy. The U-Matrix is useful for visualizing
  category separation.
- **Defect-type classification:** SOM is a useful exploratory model but not
  the strongest predictor. It exceeded GRNN on bottle, hazelnut, carpet,
  leather, tile, and wood, but RBF remained stronger overall. Its screw and
  grid results were poor because visually similar defect types overlap in
  embedding space; the SOM map makes that overlap visible.
- **Unsupervised anomaly exploration:** SOM is useful as a compact embedding
  novelty detector for bottle and tile, where its ROC-AUC was `0.983` and
  `0.978`, but it was near random for screw and grid. PatchCore remains the
  more reliable anomaly detector and is the only one of these models producing
  pixel-level heatmaps.

**Finding:** RBF was the strongest shallow classifier overall, SOM was the most
useful for object-category structure and selected anomaly-exploration cases,
and GRNN was generally weakest on these small fine-grained splits. SOM should
therefore be presented as both a classifier comparison and an interpretable
topology/novelty analysis tool, not as a replacement for PatchCore's
unsupervised anomaly localization.

To compare baseline classifier architectures, point `train_baseline.py` at any of `config/screw_config.yaml` (ResNet18), `config/screw_config_efficientnet_b0.yaml`, `config/screw_config_convnext_tiny.yaml`, `config/screw_config_vit_b_16.yaml`, or `config/screw_config_simple_cnn.yaml` — each writes its own checkpoint/metrics file so results don't overwrite each other.

To run the full pipeline on a different MVTec-AD category, copy `config/screw_config.yaml` to `config/<category>_config.yaml`, update `category` and `data.manifest_path`, then repeat steps 1–4 with `--category <category>` / `--config config/<category>_config.yaml`. Already set up this way: `screw`, `bottle`, `hazelnut`, `carpet`, `leather`, `wood`, `grid`, `tile`, `transistor` (see [Generalization to Other Categories](#generalization-to-other-categories) below). After adding a new category, retrain the category classifier so the Streamlit demo can auto-detect it too:

```bash
python -m src.models.train_category_classifier --categories screw bottle hazelnut carpet leather wood grid tile transistor <new_category>
```

## Pipeline Steps in Detail

### Step 1 — Build the manifest CSV

**Script:** [src/data/create_manifest.py](src/data/create_manifest.py)

**What it does:** walks `data/mvtec_anomaly_detection/<category>/train/good/` (all "good" images) and every `test/<defect_type>/` subfolder (`good` plus each defect type, e.g. `manipulated_front`, `scratch_head`), pairing each defective test image with its ground-truth mask from `ground_truth/<defect_type>/<name>_mask.png` when one exists. This is bookkeeping only — no model/algorithm involved, just a filesystem scan + CSV writer.

**Output:** `data/manifests/<category>.csv` with columns `image_path, mask_path, split, label, defect_type` (`label` = 0 good / 1 defective). This manifest is the single source of truth every downstream script reads from — e.g. [data/manifests/screw.csv](data/manifests/screw.csv).

### Step 2 — Train the supervised baseline classifier

**Script:** [src/models/train_baseline.py](src/models/train_baseline.py), using [src/data/dataset.py](src/data/dataset.py) (`ManifestImageDataset`, `make_train_val_split`), [src/preprocessing/transform.py](src/preprocessing/transform.py) (resize/normalize + light augmentation for train, deterministic resize/normalize for val) and [src/models/baseline_classifier.py](src/models/baseline_classifier.py) (`build_baseline_model`).

**What it does:** since MVTec's `train/` only has `good` images, this baseline instead carves a train/val split out of the *labeled* `test/` rows (`data.val_split` / `data.seed` in the config, e.g. 70/30 for screw) and trains a plain good-vs-defective image classifier on it.

**Algorithm:** one of five interchangeable architectures (`model.architecture` in the config):
- `resnet18` or `efficientnet_b0` — ImageNet-pretrained CNN backbone (transfer learning) with the final layer replaced by a 2-way linear head.
- `convnext_tiny` — an ImageNet-pretrained modern convolutional architecture that adopts design ideas popularized by vision transformers while retaining convolutional inductive biases.
- `vit_b_16` — an ImageNet-pretrained Vision Transformer that splits the 224×224 image into 16×16 patches and classifies from self-attention features.
- `simple_cnn` — a small 4-block Conv→BatchNorm→ReLU→MaxPool CNN ([SimpleCNN](src/models/baseline_classifier.py)) trained from scratch (no pretrained weights), used as a from-scratch comparison point.

Trained with cross-entropy loss and the Adam optimizer for up to `train.epochs` epochs (`train.learning_rate`, `train.batch_size` from the config). Validation loss is measured after every epoch; the weights from the lowest-validation-loss epoch are restored before final evaluation and checkpoint saving. `train.early_stopping_patience` (default 3) stops training after that many epochs without improvement, while `train.early_stopping_min_delta` controls the minimum loss decrease that counts as an improvement. Set patience to `0` to disable early stopping while still restoring the best epoch.

For a less split-sensitive estimate, add `--cross-validation --folds 5`. This runs stratified K-fold evaluation over all labeled `test/` rows, trains each fold independently with the same early-stopping rule, and reports per-fold metrics plus mean ± sample standard deviation. Cross-validation is evaluation-only: it saves out-of-fold metrics/plots but deliberately does not replace the deployable checkpoint produced by the ordinary train/validation run.

**Output** (named `baseline_<architecture>_<category>`, so different architectures don't overwrite each other):
- `models/checkpoints/baseline_<architecture>_<category>.pt` — best-validation-loss model state dict.
- `outputs/metrics/baseline_<architecture>_<category>_metrics.json` — accuracy, precision, recall, F1, confusion matrix, best epoch/loss, epochs trained, and per-epoch loss history on the held-out val subset.
- `outputs/figures/baseline_<architecture>_<category>_confusion_matrix.png` — plotted confusion matrix.
- `outputs/metrics/baseline_<architecture>_<category>_cv<folds>_metrics.json` — fold-level metrics, mean ± standard deviation, and aggregate out-of-fold metrics when cross-validation is requested.
- `outputs/figures/baseline_<architecture>_<category>_cv<folds>_confusion_matrix.png` — confusion matrix from all out-of-fold predictions.

**Classical ML alternative:** [src/models/train_pca_lda_baseline_classifier.py](src/models/train_pca_lda_baseline_classifier.py) targets the same problem/split with frozen ResNet18 embeddings → PCA → LDA instead of a fine-tuned CNN head — see [Classical ML alternative: PCA + LDA baseline classifier](#classical-ml-alternative-pca--lda-baseline-classifier) for when it does/doesn't beat the CNN classifier. Not currently wired into the Streamlit demo (which still uses the Step 2 CNN checkpoint via PatchCore/ensemble).

### Step 3 — Train and evaluate the unsupervised PatchCore anomaly detector

**Script:** [src/models/run_anomaly_detection.py](src/models/run_anomaly_detection.py), using [src/models/anomaly_detector.py](src/models/anomaly_detector.py) (`PatchCoreAnomalyDetector`), [src/preprocessing/segmentation.py](src/preprocessing/segmentation.py) (foreground mask), [src/evaluation/metrics.py](src/evaluation/metrics.py) and [src/visualization/heatmap.py](src/visualization/heatmap.py).

**What it does:** fits the detector on `train/good` only (no labels used at all), then scores every image in the full labeled `test/` split (good + every defect type).

**Algorithm — PatchCore** (Roth et al., *"Towards Total Recall in Industrial Anomaly Detection"*, CVPR 2022; this implementation keeps the core recipe and exposes the paper's optional softmax reweighting as an ablation):
1. **Locally-aware patch features:** a frozen, ImageNet-pretrained backbone (`anomaly_detection.backbone`: `resnet18` or `wide_resnet50_2`) extracts feature maps from two intermediate layers (`layer2`, `layer3`), each 3×3-average-pooled for local context and concatenated into one multi-scale patch feature map.
2. **Memory bank via greedy coreset:** all patch features from every `train/good` image are subsampled with a greedy k-center coreset algorithm (`_greedy_coreset`) — starting from a seeded patch, it repeatedly keeps the patch farthest from everything already selected — down to `anomaly_detection.coreset_ratio` of the pool, capped at `max_coreset_size` (2000) patches. Distance comparisons use a lower-dimensional projection for speed: Johnson–Lindenstrauss random projection by default, or the experimental data-dependent PCA projection when `anomaly_detection.projection_method: pca`/`--projection-method pca` is selected. The memory bank always stores the original, non-projected features.
3. **Scoring:** by default, each query patch's anomaly score is its distance to the nearest memory-bank patch (`num_neighbors: 1`). For the k-NN ablation, it is the mean of the `k` smallest distances. The image score is the maximum patch score; optional PatchCore softmax reweighting multiplies that maximum by one minus the nearest normal patch's softmax probability among its local memory-bank support. Reweighting changes only the image score, while k-NN also changes the heatmap. `compute_foreground_mask` can pin background patches to the object's minimum score before the max; the per-patch grid is bilinearly upsampled to form the heatmap.
4. **Thresholding:** the good/defective decision boundary is chosen by maximizing Youden's J statistic (`youden_threshold`) on the ROC curve of image scores over the test set.

**Evaluation** ([src/evaluation/metrics.py](src/evaluation/metrics.py)): image-level ROC-AUC + accuracy/precision/recall/F1 at the Youden threshold, plus pixel-level ROC-AUC (heatmap vs. ground-truth mask over every test pixel) and mean IoU/Dice (predicted vs. ground-truth defect mask at the pixel-level Youden threshold, defect images only).

**Output** (named `patchcore_<backbone>_<category>`):
- `models/checkpoints/patchcore_<backbone>_<category>_memory_bank.pt` — the memory bank tensor (the "trained model").
- `outputs/metrics/patchcore_<backbone>_<category>_metrics.json` — AUROC, threshold, score min/max, pixel AUROC, pixel threshold, mean IoU, mean Dice, accuracy/precision/recall/F1, projection method, and seed.
- `outputs/heatmaps/patchcore_<backbone>_<category>_<defect_type>_example.png` — one example figure per defect type: original image | predicted heatmap | ground-truth mask | overlay, for a quick visual sanity check.

Experimental overrides append identifiers instead of overwriting the default artifacts: PCA uses `_pcaproj`, an explicit/configured seed uses `_seed<seed>`, k-NN uses `_knn<k>`, and reweighting uses `_rw<support>` (for example, `patchcore_resnet18_screw_knn3_rw9_metrics.json`). `--metrics-only` still evaluates and saves the JSON metrics but skips the memory-bank checkpoint and example heatmaps, which is useful for multi-run benchmarks.

### Step 4 — Train the category classifier (auto-detect object type)

**Script:** [src/models/train_category_classifier.py](src/models/train_category_classifier.py).

**What it does:** combines every requested category's manifest (train + test rows, good and defective alike — object-type recognition doesn't care about defect status), replaces the good/defective label with a category index, and trains a multi-class "which object is this?" classifier on a stratified train/val split.

**Algorithm:** ImageNet-pretrained ResNet18 (`build_baseline_model`) with an `N`-way head (`N` = number of categories), trained with cross-entropy + Adam for 8 epochs by default.

**Output:**
- `models/checkpoints/category_classifier_resnet18.pt` — model state dict (best-validation-loss epoch, restored before saving).
- `outputs/metrics/category_classifier_metrics.json` — the ordered category list (defines the label→name mapping), validation accuracy, confusion matrix, per-class classification report, per-epoch history, seed, and best epoch.
- `outputs/figures/category_classifier_resnet18_training_curves.png` — per-epoch train/val loss and accuracy curves.

Training is seeded (`--seed`, default 42), evaluates on the validation split every epoch, restores the lowest-val-loss epoch's weights before saving, and supports patience-based early stopping (`--early-stopping-patience`, default 3, `0` to disable) — the same controls as the Step 2 baseline trainer.

### Step 5 — Train the defect-type classifier (per category)

**Script:** [src/models/train_defect_classifier.py](src/models/train_defect_classifier.py).

**What it does:** unlike the Step 2 good/defective classifier, this only looks at a single category's *defective* test/ rows and predicts which `defect_type` it is (e.g. leather: `color`/`cut`/`fold`/`glue`/`poke`), on a stratified train/val split over just those defect types.

**Algorithm:** the same `build_baseline_model` architecture as the category's `config/<category>_config.yaml` (ResNet18 by default), with an `N`-way head (`N` = number of defect types for that category), trained with cross-entropy + Adam for up to 10 epochs. Training seeds NumPy/PyTorch from `data.seed`, evaluates on the validation split every epoch, restores the lowest-val-loss epoch's weights before saving, and early-stops after `train.early_stopping_patience` (default 3) epochs without improvement. `--freeze-backbone` freezes everything except the classification head (~2k trainable parameters instead of ~11M) as an overfitting control, `--learning-rate` overrides the config's LR (a frozen head needs a much higher LR than fine-tuning), and `--epochs` overrides the config's epoch budget (useful to confirm val loss has actually plateaued rather than just hitting the default epoch count).

**Output** (named `defect_classifier_<architecture>_<category>`):
- `models/checkpoints/defect_classifier_<architecture>_<category>.pt` — model state dict (best-validation-loss epoch).
- `outputs/metrics/defect_classifier_<architecture>_<category>_metrics.json` — the ordered defect-type list (defines the label→name mapping), validation accuracy, confusion matrix, per-class classification report, per-epoch history, seed, and best epoch.
- `outputs/figures/defect_classifier_<architecture>_<category>_training_curves.png` — per-epoch train/val loss and accuracy curves.
- Focused variants append `_focused` (ground-truth oracle) or `_focused_patchcore` (deployment-matched crops); `--freeze-backbone` appends `_frozen`. Metrics record crop mode/source, padding, and minimum crop fraction.
7. If "Defective" and a Step 5 defect-type classifier exists for the category, runs it to show the predicted **defect type** and its confidence alongside the verdict. When `defect_classifier.use_focused_crops: true`, it crops around the PatchCore anomaly map at the stored pixel threshold first; otherwise it uses the full image. Missing focused artifacts fall back to the full-image checkpoint.

This is optional per category — the Streamlit demo checks whether a checkpoint/metrics file exists for the detected category and simply skips the defect-type display (with a hint to train it) if not.

**Classical ML alternative:** [src/models/train_pca_lda_defect_classifier.py](src/models/train_pca_lda_defect_classifier.py) targets the same problem/split with frozen ResNet18 embeddings → PCA → LDA instead of a fine-tuned CNN head — see [Classical ML alternative: PCA + LDA defect-type classifier](#classical-ml-alternative-pca--lda-defect-type-classifier) for when it does/doesn't beat the CNN head. Not currently wired into the Streamlit demo (which still uses the Step 5 CNN checkpoint).

### Step 6 — Launch the interactive Streamlit demo

**Script:** [app/streamlit_app.py](app/streamlit_app.py).

**What it does, end to end, for an uploaded image:**
1. Runs the Step 4 category classifier to auto-detect the object type (`screw`/`bottle`/`hazelnut`/`carpet`/`leather`/`wood`/`grid`/`tile`/`transistor`), with a collapsed manual-override dropdown as a fallback.
2. Loads that category's config, PatchCore memory bank (Step 3 checkpoint) and metrics file (for the decision threshold).
3. Computes the Otsu foreground mask (`compute_foreground_mask`) unless the category's config sets `use_foreground_mask: false` (full-frame textures like `bottle`/`carpet`/`leather`/`wood`/`grid`/`tile`, or a discrete object on a busy non-uniform background like `transistor`).
4. Runs `PatchCoreAnomalyDetector.predict` to get the image anomaly score and pixel-level anomaly map.
5. Turns the score into a **Normal / Defective** verdict by comparing against the stored Youden threshold.
6. Buckets a "Defective" verdict into **Low / Medium / High severity** (`classify_severity`) based on what *fraction of the object's foreground area* is above the threshold, not just the raw score.
7. If "Defective" and a Step 5 defect-type classifier exists for the category, runs it to show the predicted **defect type** and its confidence alongside the verdict.
8. Builds a threshold-anchored jet-colormap heatmap and image+heatmap overlay (`make_overlay` / `normalize_map_threshold`) so warm colors visually agree with the Normal/Defective decision.

**Output:** an interactive UI showing the original image, the anomaly heatmap, and the overlay side by side, plus Prediction / Anomaly score / Severity / (optional) Defect type metrics — nothing is persisted to disk (aside from Streamlit's in-memory model cache).

### Step 7 (optional) — Fuse the classifier and PatchCore into a hybrid ensemble

**Script:** [src/models/run_ensemble.py](src/models/run_ensemble.py) — see [Hybrid Ensemble](#hybrid-ensemble-fusing-the-classifier-and-patchcore) below for the full write-up, algorithm (weighted-average score fusion) and output (`outputs/metrics/ensemble_<architecture>_<category>_metrics.json`).

### Step 8 — Run the unit tests

**Command:** `python -m pytest tests/ -v`

**What's tested:** pure evaluation/preprocessing/visualization functions plus baseline-training control flow; no trained checkpoints are required —
- [tests/test_metrics.py](tests/test_metrics.py): `compute_classification_metrics`, `youden_threshold`, `compute_iou`/`compute_dice`, `compute_pixel_level_metrics` against hand-built synthetic labels/scores/masks.
- [tests/test_heatmap.py](tests/test_heatmap.py): `normalize_map`/`normalize_map_threshold`/`make_overlay` produce correctly-shaped, correctly-ranged outputs.
- [tests/test_segmentation.py](tests/test_segmentation.py): `compute_foreground_mask` correctly separates a synthetic bright object from a dark background (and vice versa).
- [tests/test_train_baseline.py](tests/test_train_baseline.py): early stopping restores the best epoch's weights and invalid K-fold settings are rejected before training.
- [tests/test_projection_benchmark.py](tests/test_projection_benchmark.py): seeded projection runs are aggregated with the expected mean, sample standard deviation, and delta from the random-projection baseline.
- [tests/test_anomaly_runner.py](tests/test_anomaly_runner.py): default and tuned PatchCore artifact names remain distinct for CLI- and config-defined settings.
- [tests/test_patchcore_tuning.py](tests/test_patchcore_tuning.py): tuning runs are ranked correctly and identical configurations are aggregated across seeds with mean ± sample standard deviation.
- [tests/test_defect_crop.py](tests/test_defect_crop.py): focused crop boxes are square, padded/clamped correctly, resize mask coordinates safely, fall back for empty masks, and are applied by the manifest dataset before transforms.
- [tests/test_anomaly_scoring.py](tests/test_anomaly_scoring.py): k-NN means/indices, PatchCore softmax weights, invalid settings, artifact suffixes, and the detector-level contract that reweighting changes only image scores.
- [tests/test_patchcore_scoring_summary.py](tests/test_patchcore_scoring_summary.py): scoring experiments are scoped to the correct baseline, ranked, and compared with metric deltas.

**Output:** pytest pass/fail report in the terminal; no files are written.

## Model Comparison (Screw Category)

Same train/val split (112/48 images carved from `test/`), same maximum 10 epochs, same optimizer/learning rate — only the architecture changes. These rows are the existing single-split architecture comparison; they are not K-fold results.

| Model | Type | Params | Pretrained | Final train loss | Val accuracy / precision / recall / F1 |
|---|---|---|---|---|---|
| ResNet18 | Transfer learning (CNN) | 11.2M | ✅ ImageNet | 0.010 | 1.00 / 1.00 / 1.00 / 1.00 |
| EfficientNet-B0 | Transfer learning (CNN) | 4.0M | ✅ ImageNet | 0.041 | 1.00 / 1.00 / 1.00 / 1.00 |
| ConvNeXt-Tiny | Transfer learning (modern CNN) | 27.8M | ✅ ImageNet | **0.002** | 1.00 / 1.00 / 1.00 / 1.00 |
| ViT-B/16 | Transfer learning (vision transformer) | 85.8M | ✅ ImageNet | 0.081 | 1.00 / 1.00 / 1.00 / 1.00 |
| Simple CNN (sequential) | From-scratch CNN | 0.25M | ❌ | 0.175 | 1.00 / 1.00 / 1.00 / 1.00 |
| PatchCore (ResNet18 features) | Unsupervised anomaly detection | — | ✅ ImageNet (frozen) | n/a | Image ROC-AUC 0.91, Pixel ROC-AUC 0.98 |

**Finding:** all five supervised classifiers hit the same perfect validation score, because (per the lesson below) the validation split is tiny and every defect type in it was already seen during training — accuracy/F1 cannot distinguish them. The **training loss curve and resource cost are more informative signals**: ConvNeXt-Tiny reached the lowest final loss (0.0024), followed by ResNet18 (0.010), EfficientNet-B0 (0.041), ViT-B/16 (0.081), and the from-scratch Simple CNN (0.175). ViT's loss was also less stable (including a spike to 1.057 at epoch 4) and it is by far the largest model at 85.8M parameters; on only 112 training images, its weaker image-locality inductive bias and higher capacity offer no measurable validation benefit over the CNNs. ConvNeXt is a strong modern-CNN result, but its 27.8M parameters likewise buy no validation-score gain over the much smaller 4.0M EfficientNet-B0. **Recommendation:** keep EfficientNet-B0 or ResNet18 as the practical supervised baseline, use ConvNeXt-Tiny as the strongest-convergence architecture comparison, and treat ViT-B/16 as an educational architecture ablation rather than a performance upgrade on this tiny dataset. PatchCore remains the more trustworthy detector because it does not depend on the supervised model's leaky validation setup.

**Cross-validation status:** stratified K-fold execution, out-of-fold aggregation, and best-epoch restoration are implemented and covered by tests, but a complete five-fold architecture benchmark has not yet been run or added to this table. Generate it per architecture with `--cross-validation --folds 5`; compare the reported mean ± standard deviation rather than a single fold's score.

### PatchCore backbone: ResNet18 vs WideResNet50-2

Same memory bank size (2000 patches), same coreset/foreground-masking settings — only the frozen feature-extractor backbone changes (`config/screw_config_wide_resnet50_2.yaml`).

| Backbone | Image ROC-AUC | Pixel ROC-AUC | Mean IoU | Mean Dice |
|---|---|---|---|---|
| ResNet18 | 0.909 | 0.976 | 0.047 | 0.088 |
| WideResNet50-2 | **0.935** | **0.978** | 0.044 | 0.083 |

**Finding:** the larger WideResNet50-2 backbone gives richer patch features, improving image-level ROC-AUC by ~2.6 points (0.909 → 0.935) and pixel-level ROC-AUC slightly, at the cost of a much larger download/forward pass (~264MB vs ~45MB, noticeably slower per image on CPU). Mean IoU/Dice don't improve — both backbones extract features at the same 28×28 spatial grid, so the blur from bilinear upsampling (the actual limiting factor for tight segmentation, per the lesson below) is unaffected by backbone size. **Recommendation:** use WideResNet50-2 when detection accuracy matters more than latency/footprint (e.g. batch QA review); keep ResNet18 for fast/interactive use (e.g. the Streamlit demo).

### PatchCore parameter tuning: coreset size, projection dimension, and feature layers

[src/models/run_anomaly_detection.py](src/models/run_anomaly_detection.py) accepts `--max-coreset-size`, `--projection-dim`, and one-or-more `--layers` overrides. Metrics record the effective settings and memory-bank size; artifact names include `_cs<size>`, `_pd<dim>`, and the layer names so experiments cannot overwrite the default run. [src/evaluation/summarize_patchcore_tuning.py](src/evaluation/summarize_patchcore_tuning.py) discovers these runs, computes deltas from the default, ranks configurations by image AUROC, and aggregates repeated settings across seeds.

The initial screw sweep changed one factor at a time with ResNet18, random projection, seed 42, and the same preprocessing. The baseline uses a 2,000-patch cap, 128-dimensional projection, and `layer2+layer3` features:

| Configuration | Image AUROC | Pixel AUROC | Mean IoU | Mean Dice |
|---|---:|---:|---:|---:|
| Default: 2,000 / 128 / layer2+layer3 | 0.9092 | 0.9761 | 0.0471 | 0.0884 |
| Coreset 1,000 | 0.8664 | 0.9714 | 0.0345 | 0.0660 |
| Coreset 4,000 | 0.9395 | **0.9794** | 0.0488 | 0.0916 |
| Projection dim 64 | 0.9174 | 0.9770 | 0.0351 | 0.0671 |
| Projection dim 256 | 0.9484 | 0.9765 | 0.0365 | 0.0697 |
| layer2 only | 0.9436 | 0.9778 | 0.0491 | 0.0920 |
| layer3 only | 0.8000 | 0.9709 | 0.0270 | 0.0520 |
| layer1+layer2 | **0.9498** | 0.9704 | **0.0553** | **0.1031** |

The independently successful settings composed well. A 4,000-patch bank, 256-dimensional projection, and `layer2` features were repeated at seeds 41/42/43:

| Configuration | Image AUROC | Pixel AUROC | Mean IoU | Mean Dice |
|---|---:|---:|---:|---:|
| Default (seed 42) | 0.9092 | 0.9761 | 0.0471 | 0.0884 |
| Tuned, seeds 41/42/43 | **0.9696 ± 0.0061** | **0.9808 ± 0.0009** | **0.0677 ± 0.0030** | **0.1237 ± 0.0048** |

```bash
# Reproduce the one-factor experiments (examples)
python -m src.models.run_anomaly_detection --config config/screw_config.yaml --metrics-only --max-coreset-size 4000
python -m src.models.run_anomaly_detection --config config/screw_config.yaml --metrics-only --projection-dim 256
python -m src.models.run_anomaly_detection --config config/screw_config.yaml --metrics-only --layers layer2

# Run the selected screw configuration with full checkpoint/heatmap output
python -m src.models.run_anomaly_detection --config config/screw_config_tuned_patchcore.yaml

# Aggregate all tuning metrics
python -m src.evaluation.summarize_patchcore_tuning --category screw
```

**Finding:** reducing the memory bank or using only the deeper, coarser `layer3` features loses substantial information. A larger bank preserves more normal variation, a wider random projection preserves coreset-selection distances better, and `layer2` retains the finer spatial/detail signal most useful for screw defects. The combined setting improves every reported metric across three seeds, but costs roughly twice the memory-bank storage and nearest-neighbor scoring work versus 2,000 patches; fitting also becomes slower. Keep [config/screw_config.yaml](config/screw_config.yaml) as the lightweight cross-category baseline and use [config/screw_config_tuned_patchcore.yaml](config/screw_config_tuned_patchcore.yaml) when screw accuracy matters more than latency. These values were tuned on screw and must not be copied to other categories without category-specific, multi-seed validation.

### PatchCore scoring: k-NN and softmax reweighting

[src/models/anomaly_detector.py](src/models/anomaly_detector.py) supports two optional scoring changes. With `--num-neighbors k`, patch score becomes the mean distance to the `k` nearest memory-bank patches instead of the single nearest distance. With `--softmax-reweighting`, only the maximum image-level score is adjusted using the nearest normal patch's neighborhood (9 support patches by default); the anomaly map remains the ordinary nearest-neighbor map unless k-NN is also enabled.

```bash
# Isolate each scoring change on the category's default detector
python -m src.models.run_anomaly_detection --config config/screw_config.yaml \
  --num-neighbors 3 --metrics-only
python -m src.models.run_anomaly_detection --config config/screw_config.yaml \
  --softmax-reweighting --metrics-only

# Compare scoring runs with their matching baseline
python -m src.evaluation.summarize_patchcore_scoring --category screw
```

Same ResNet18/default coreset and seed 42; only scoring changes. Values show baseline → k=3, followed by the 9-support softmax image AUROC (softmax cannot change pixel metrics):

| Category | Image AUROC | Pixel AUROC | Mean IoU | Mean Dice | Softmax image AUROC |
|---|---:|---:|---:|---:|---:|
| Screw | 0.9092 → **0.9389** | 0.9761 → **0.9783** | 0.0471 → **0.0574** | 0.0884 → **0.1061** | 0.9043 |
| Bottle | 1.0000 → 1.0000 | 0.9813 → 0.9814 | **0.3966** → 0.3952 | **0.5505** → 0.5487 | 1.0000 |
| Hazelnut | 0.9982 → **0.9993** | 0.9699 → **0.9719** | 0.2093 → **0.2181** | 0.3196 → **0.3313** | 0.9979 |
| Carpet | 0.9711 → **0.9767** | 0.9874 → **0.9879** | **0.2512** → 0.2465 | **0.3716** → 0.3666 | 0.9715 |
| Leather | 1.0000 → 1.0000 | 0.9911 → **0.9913** | 0.1265 → **0.1357** | 0.2097 → **0.2225** | 1.0000 |

On the stronger item-3 screw configuration (4,000 patches, projection 256, `layer2`), paired seeds 41/42/43 show the tradeoff more clearly:

| Scoring | Image AUROC | Pixel AUROC | Mean IoU | Mean Dice |
|---|---:|---:|---:|---:|
| k=1 | **0.9696 ± 0.0061** | 0.9808 ± 0.0009 | **0.0677 ± 0.0030** | **0.1237 ± 0.0048** |
| k=3 | 0.9659 ± 0.0031 | **0.9820 ± 0.0008** | 0.0673 ± 0.0050 | 0.1229 ± 0.0081 |

**Finding:** k=3 is useful on the weaker default detector and improves at least one ranking metric in all five categories, with meaningful screw/hazelnut/leather localization gains. It is not additive with every prior optimization: on the tuned screw detector it improves pixel ranking but slightly lowers mean image AUROC and overlap. k=5 improves default screw image AUROC further to 0.9469 but gives back the k=3 overlap gains (IoU/Dice 0.0511/0.0953). Softmax reweighting is not beneficial here: on non-ceiling screw/hazelnut it lowers image AUROC, on carpet it adds only 0.0004, and bottle/leather are unchanged at 1.0. **Recommendation:** keep k=1 and softmax disabled as global defaults; use k=3 as a per-category option when its measured localization/ranking tradeoff is desired. Do not enable softmax reweighting for the current categories.

### PatchCore coreset projection: random (JL) vs PCA — an ablation

[src/models/anomaly_detector.py](src/models/anomaly_detector.py)'s greedy coreset selection only ever uses a projection to speed up the pairwise-distance computation that picks which patches to keep — the memory bank itself always stores the original, non-projected features. The default (and the PatchCore paper's choice) is a Johnson–Lindenstrauss **random** projection; `anomaly_detection.projection_method: pca` (added for this experiment, see `config/screw_config_pca_projection.yaml`) swaps it for a **PCA** projection (`torch.pca_lowrank`) onto the same `projection_dim`, fit on the pooled train/good patches themselves, fully data-dependent instead of random.

```bash
python -m src.models.run_anomaly_detection --config config/screw_config_pca_projection.yaml
```

For the cross-category, multi-seed benchmark, use explicit overrides so every metrics file has a unique seed suffix. `--metrics-only` avoids writing a redundant memory bank and example heatmaps for every experimental run:

```bash
for category in screw bottle hazelnut carpet leather; do
  for seed in 41 42 43; do
    python -m src.models.run_anomaly_detection \
      --config "config/${category}_config.yaml" \
      --projection-method pca \
      --seed "$seed" \
      --metrics-only
  done
done

python -m src.evaluation.summarize_projection_benchmark \
  --categories screw bottle hazelnut carpet leather \
  --seeds 41 42 43
```

The summary is written to `outputs/metrics/patchcore_resnet18_pcaproj_multiseed_summary.json`. Each category reports mean ± sample standard deviation across PCA seeds and the delta from its existing default random-projection run (seed 42).

Same backbone (ResNet18), memory bank size (2000 patches), projection dimension (128), and per-category foreground-masking settings — only the coreset's internal projection/seed changes. PCA values are mean ± sample standard deviation over seeds 41/42/43; the random baseline is the existing seed-42 run:

| Category | Random image AUROC | PCA image AUROC | Δ | Random pixel AUROC | PCA pixel AUROC | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Screw | 0.9092 | **0.9212 ± 0.0144** | +0.0120 | 0.9761 | **0.9784 ± 0.0004** | +0.0024 |
| Bottle | **1.0000** | 1.0000 ± 0.0000 | 0.0000 | 0.9813 | **0.9813 ± 0.0000** | +0.0000 |
| Hazelnut | 0.9982 | **0.9998 ± 0.0002** | +0.0015 | **0.9699** | 0.9696 ± 0.0009 | -0.0004 |
| Carpet | 0.9711 | **0.9720 ± 0.0030** | +0.0009 | **0.9874** | 0.9870 ± 0.0005 | -0.0004 |
| Leather | **1.0000** | 1.0000 ± 0.0000 | 0.0000 | 0.9911 | **0.9912 ± 0.0001** | +0.0000 |

**Finding — PCA is category- and seed-dependent, not a general PatchCore upgrade.** Screw is the only category with a material average image-level gain (+1.20 points), but its variability is large (0.9212 ± 0.0144): seed 42 produced the earlier headline 0.9356, while seed 43 fell to 0.9067 — below the 0.9092 random baseline. Averaging the seeds also reverses the earlier localization claim for screw: mean IoU/Dice are 0.0451/0.0851 versus random's 0.0471/0.0884. Bottle and leather are already at the image-AUROC ceiling, and PCA changes essentially nothing. Hazelnut/carpet gain only 0.15/0.09 image-AUROC points while pixel AUROC decreases by 0.04 points; their mean IoU/Dice nevertheless improve (hazelnut 0.209/0.320 → 0.216/0.329, carpet 0.251/0.372 → 0.260/0.381), illustrating again that ranking and thresholded overlap are separate axes. **Recommendation:** keep random projection as the global default; treat PCA as a per-category option worth considering for screw after multiple-seed validation, not as a universal replacement. A fairer future comparison should also run the random projection over the same seeds instead of comparing a PCA distribution only against one random seed.

## Hybrid Ensemble: Fusing the Classifier and PatchCore

[src/models/run_ensemble.py](src/models/run_ensemble.py) fuses the supervised classifier's softmax "defective" probability with PatchCore's (min-max normalized) anomaly score into one weighted-average score, and evaluates classifier-only, PatchCore-only, and the fused ensemble side by side:

```bash
python -m src.models.run_ensemble --config config/screw_config.yaml --classifier-weight 0.5
```

**Important methodological note:** this is evaluated only on the classifier's held-out validation subset (48 images) — the *only* data the classifier hasn't been fit on. PatchCore, by contrast, has never seen *any* test-set image during training, so it's evaluated fairly on all 160 test images elsewhere in this README; on this smaller 48-image slice alone it scores a lower 0.884 AUROC (accuracy 0.79) than its full-test-set 0.909, simply due to the smaller/different sample.

| Signal | AUROC (on the 48-image held-out subset) | Accuracy |
|---|---|---|
| Classifier only | 1.00 | 1.00 |
| PatchCore only | 0.884 | 0.79 |
| Fused ensemble (0.5 / 0.5 and 0.3 / 0.7 weights) | 1.00 | 1.00 |

**Finding:** the fused ensemble ties the classifier alone rather than clearly beating it, at both a 50/50 and a 30/70 (classifier/PatchCore) weighting. This isn't a failure of the fusion code — it's the same root cause documented below: the classifier's own held-out subset is still drawn from `test/`, where every defect *type* was already seen during its training subset, so its predictions are already perfectly separable there and there's no ceiling left for the ensemble to break through. The ensemble machinery is correctly implemented and doesn't hurt anything, but demonstrating its real value would need a genuinely novel, held-out defect sample not derived from `test/` at all — which this dataset's split (`train/`=good only, `test/`=only labeled data) doesn't provide.

## Generalization to Other Categories

The same pipeline (manifest → baseline classifier → PatchCore → Streamlit demo) was run end-to-end on eight more MVTec-AD categories, picked to be different in shape: `bottle` (top-down shot of a bottle mouth), `hazelnut` (small object on a plain background, closer to `screw`), `carpet` (a close-up textile texture filling the whole frame, no discrete object at all), `leather` (another close-up, full-frame texture, same situation as carpet), `wood` (wood grain texture surface), `grid` (regular geometric grid pattern), `tile` (repeating tile texture), and `transistor` (a small discrete object like `screw`/`hazelnut`, but sitting on a busy, non-uniform perforated circuit-board background instead of a plain one).

| Category | Classifier val accuracy/F1 | PatchCore image ROC-AUC | PatchCore pixel ROC-AUC | Mean IoU | Mean Dice |
|---|---|---|---|---|---|
| Screw | 1.00 / 1.00 | 0.909 | 0.976 | 0.047 | 0.088 |
| Bottle | 0.88 / 0.92 | 1.000 | 0.981 | 0.397 | 0.550 |
| Hazelnut | 1.00 / 1.00 | 0.998 | 0.970 | 0.209 | 0.320 |
| Carpet | 0.92 / 0.94 | 0.971 | 0.987 | 0.251 | 0.372 |
| Leather | 1.00 / 1.00 | 1.000 | **0.991** | 0.127 | 0.210 |
| Grid | 0.83 / 0.88 | 0.784 | 0.947 | 0.076 | 0.135 |
| Tile | 0.94 / 0.96 | **0.997** | 0.941 | 0.383 | 0.518 |
| Wood | 1.00 / 1.00 | 0.989 | 0.930 | 0.187 | 0.291 |
| Transistor | 0.93 / 0.91 | **0.998** | 0.961 | 0.248 | 0.351 |

**Finding — the classifier's "perfect scores" issue isn't universal.** Unlike screw and hazelnut, the bottle classifier scored a believable 0.88 accuracy / 0.92 F1, not 1.0 (its held-out val subset is smaller — only 25 images — and the defects are more subtle). This is a useful counter-example confirming that the earlier "misleadingly perfect" finding is specifically a symptom of the *screw* dataset being small/easy, not a bug in the evaluation code.

**Finding — the foreground-masking heuristic doesn't generalize automatically, and blindly applying it can actively hurt localization.** The Otsu-based foreground mask ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py)) assumes a plain background with the object as the minority of pixels — true for screw and hazelnut, but **false for bottle**, whose images are a top-down shot where the bottle mouth fills the entire frame. Applying it anyway made bottle's pixel-level ROC-AUC **worse than random (0.374)**: Otsu split the frame into the dark inner bottle opening vs. the lighter rim, and incorrectly zeroed out real defect pixels that happened to fall inside the dark "background" region. Adding a `use_foreground_mask: false` toggle to `bottle_config.yaml` (and threading it through [run_anomaly_detection.py](src/models/run_anomaly_detection.py) and the Streamlit app) fixed it immediately: pixel ROC-AUC jumped to 0.981 and mean IoU/Dice became the *best* of screw/bottle/hazelnut (0.40 / 0.55) — confirmed visually, the predicted heatmap now matches the crescent-shaped ground-truth defect almost exactly. **Lesson:** any hand-crafted heuristic derived from one category's visual layout should be treated as a per-category, config-driven option, not a hardcoded assumption — and always sanity-check pixel-level metrics per category rather than assuming an improvement that helped one category will help (or even be neutral for) another.

**Applying the lesson upfront — `carpet` and `leather`.** Both are full-frame textures just like `bottle` (no discrete object vs. background), so their configs were created with `use_foreground_mask: false` from the start instead of discovering the problem the hard way again. Result: carpet got the best pixel-level ROC-AUC of the first four categories (0.987), and leather pushed that further to **0.991** — the best across the five-category run — on the first run each time, concrete evidence that the earlier fix generalized into a repeatable, config-driven decision rather than a one-off patch. Leather's mean IoU/Dice (0.127 / 0.210) are lower than carpet's, though — a reminder that pixel ROC-AUC (ranking) and IoU/Dice (tight overlap) are still independent axes even within the same `use_foreground_mask: false` texture group (see the IoU/Dice lesson below).

**Scaling to eight categories — `wood`, `grid`, and `tile`.** Adding three more texture-based categories pushed the analysis to eight total. Grid (a regular geometric pattern) shows weaker performance overall (0.784 image AUROC, 0.83 classifier accuracy) — likely due to the regular repeating pattern making subtle defect anomalies harder to distinguish from normal pattern variations. Tile (a repeating tile texture) performs strongly, nearly matching leather's performance (0.997 image AUROC, 0.94 classifier accuracy) with the best mean IoU/Dice (0.383 / 0.518) among all eight categories, suggesting its defects are spatially localized and visually distinct. Wood (wood grain texture) achieves perfect classifier accuracy (1.00) and strong PatchCore image AUROC (0.989), but lower pixel-level AUROC (0.930) and IoU/Dice, indicating good image-level anomaly detection but less precise pixel localization — likely because wood grain itself has high natural texture variation. All three new categories maintained the eight-category category classifier's 100% validation accuracy, confirming that the object types remain visually distinct even as the dataset scales.

**A new twist — `transistor` is a discrete object but still needs masking disabled.** Unlike carpet/leather, `transistor` *does* have a clear discrete object (a small transistor package with metal leads) sitting on a minority of the frame — the same layout as screw/hazelnut, where the Otsu mask helps. But its background is a perforated copper board (alternating light copper strips and dark mounting holes), not a plain surface, so Otsu's plain-background assumption still doesn't hold: the dark holes score similarly to the dark transistor body, and would get pulled into the "foreground" alongside it. `transistor_config.yaml` was created with `use_foreground_mask: false` for this reason, and the result is the **best PatchCore image ROC-AUC of any category so far (0.998, tied with hazelnut)** with a solidly mid-pack pixel ROC-AUC (0.961) and IoU/Dice (0.248 / 0.351) — in the same range as bottle/hazelnut/carpet, not the worst-case seen with bottle's original broken masking. **Lesson:** the foreground-mask decision isn't just "discrete object vs. full-frame texture" — it's really "is the background plain/uniform", and a discrete-object category can still need masking disabled if its background is visually busy.

The Streamlit demo ([app/streamlit_app.py](app/streamlit_app.py)) doesn't require the user to pick a category at all: [src/models/train_category_classifier.py](src/models/train_category_classifier.py) trains a small ResNet18 classifier to recognize the object type itself — every category classifier trained so far has hit 100% validation accuracy, whether on the eight-category set (carpet, leather, wood, grid, and tile added) or the separate six-category set with `transistor` added — and the app runs it first on the uploaded image to auto-detect the category, then routes to that category's PatchCore detector automatically — with a collapsed "override" dropdown as a manual fallback if it's ever wrong. Adding a new category only requires a new `config/<category>_config.yaml` entry in `CATEGORY_CONFIGS` plus retraining the category classifier with it included; the two extensions above haven't yet been retrained together as one combined nine-category classifier.

### Classical ML alternative: PCA + LDA baseline classifier

[src/models/train_pca_lda_baseline_classifier.py](src/models/train_pca_lda_baseline_classifier.py) applies the same classical pipeline as the [defect-type classifier's PCA+LDA alternative](#classical-ml-alternative-pca--lda-defect-type-classifier) (frozen ResNet18 embeddings → `StandardScaler` → PCA (30 components) → LDA) to the coarser good-vs-defective problem instead, on the exact same `make_train_val_split` split as [train_baseline.py](src/models/train_baseline.py):

```bash
python -m src.models.train_pca_lda_baseline_classifier --config config/screw_config.yaml
```

| Category | Deep classifier (CNN, ResNet18) val accuracy / F1 | PCA+LDA val accuracy / F1 |
|---|---|---|
| Screw | **1.00 / 1.00** | 0.917 / 0.946 |
| Bottle | 0.88 / 0.92 | **0.960 / 0.974** |
| Hazelnut | **1.00 / 1.00** | 0.848 / 0.878 |
| Carpet | **0.92 / 0.94** | 0.889 / 0.926 |
| Leather | **1.00 / 1.00** | 0.921 / 0.945 |

**Finding — the exact same pattern as the defect-type classifier repeats here.** PCA+LDA only beats the deep classifier on bottle — the one category where the deep classifier itself was weakest (0.88 accuracy, smallest val subset at 25 images) — and is worse everywhere else, most on hazelnut (1.00 → 0.848). This reproduces, on an independent classifier/problem, the same conclusion reached for defect-type classification: a lower-capacity classical classifier on frozen embeddings is a good bet specifically when the deep model is struggling (usually from too little/too-hard-to-learn-from data), not a general replacement for it. See the [Notes & Lessons Learned](#notes--lessons-learned) entry below for the combined takeaway across both classifiers.

## Defect-Type Classification (per category)

Beyond the binary Normal/Defective verdict, [src/models/train_defect_classifier.py](src/models/train_defect_classifier.py) trains a per-category, multi-class classifier over each category's own `defect_type` labels (defective images only, see [Step 5](#step-5--train-the-defect-type-classifier-per-category)), and [app/streamlit_app.py](app/streamlit_app.py) shows the predicted defect type + confidence whenever a "Defective" verdict is reached and a matching checkpoint exists.

**Note:** The train/val image counts below refer only to **defective test images** (label=1), not the full test set. For example, screw has 160 total test images (41 good + 119 defective), but the defect-type classifier only uses the 119 defective ones, split 70/30 into training and validation.

| Category | Defect types | Train / val images (defective only) | Val accuracy |
|---|---|---|---|
| Screw | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top (5) | 83 / 36 | 0.444 |
| Bottle | broken_large, broken_small, contamination (3) | 44 / 19 | 0.789 |
| Hazelnut | crack, cut, hole, print (4) | 49 / 21 | 0.857 |
| Carpet | color, cut, hole, metal_contamination, thread (5) | 62 / 27 | 0.815 |
| Leather | color, cut, fold, glue, poke (5) | 64 / 28 | **0.964** |
| Grid | bent, broken, glue, metal_contamination, thread (5) | 40 / 17 | 0.222 |
| Tile | crack, glue_strip, gray_stroke, oil, rough (5) | 59 / 25 | **0.923** |
| Wood | color, combined, hole, liquid, scratch (5) | 42 / 18 | 0.667 |
| Transistor | bent_lead, cut_lead, damaged_case, misplaced (4) | 28 / 12 | 0.750 |

**Finding — fine-grained defect-type accuracy tracks per-class sample count, not just class count.** Screw and leather both have 5 defect types, yet screw's val accuracy (0.444) is far worse while leather's (0.964) is among the best. Screw's confusion matrix shows `thread_side` and `thread_top` absorbing most misclassifications — with only 83 train images spread over 5 classes (~16–17 images/class), and screw's defect types being subtle, visually-similar deviations on the same small grey object (a scratch on the head vs. the neck, thread wear on one side vs. the top), there's neither enough data nor enough visual separation for reliable distinction. Leather's defect types, by contrast, are visually distinct surface phenomena (a color blotch vs. a cut vs. a glue smear) despite a similarly small dataset (64 train images), so it reaches near-perfect accuracy.

**Scaling to eight categories — grid, tile, and wood.** Adding three more defect-type classifiers reveals the wide variance in fine-grained accuracy even within a balanced 5-defect-type setup. **Grid** performs poorly (0.222 accuracy) — its defect types (bent, broken, glue, metal_contamination, thread) are subtle spatial/material variations on a regular geometric grid pattern, making them inherently confusable (the confusion matrix shows glue absorbing predictions from all other classes). **Tile** achieves excellent performance (0.923 accuracy, the second-best across all eight categories) — its defect types (crack, glue_strip, gray_stroke, oil, rough) are visually distinct surface phenomena (structural damage vs. surface contamination/discoloration), providing clear visual separation despite similar dataset size to grid. **Wood** reaches a middle ground (0.667 accuracy) — its defect types (color, combined, hole, liquid, scratch) include some visually distinct types (hole vs. scratch) but combined defects and color blotches are harder to disentangle from natural wood grain variation.

**Lesson:** unlike the good/defective and category classifiers, a fine-grained defect-type classifier's accuracy depends heavily on how visually distinguishable that category's specific defect types are from each other, not just on how many classes or how many total images there are. Screw (0.444) and grid (0.222) are cases where the defect types are inherently subtle or overlapping even with more training data; leather (0.964) and tile (0.923) succeed because their defect types are visually distinct phenomena. The task is fundamentally harder than object-category detection (100% accuracy across all eight categories) or binary good/defective classification (both are coarser-grained decisions with better natural class separation).

### Defect-focused crop ablation

[src/preprocessing/defect_crop.py](src/preprocessing/defect_crop.py) converts a binary localization mask into a padded square crop. Training can use either MVTec ground-truth masks (`--crop-source ground-truth`) or the default PatchCore detector's predicted pixel mask (`--crop-source patchcore`). The latter matches deployment: [app/streamlit_app.py](app/streamlit_app.py) thresholds the uploaded image's PatchCore anomaly map, intersects it with the object foreground when enabled, applies the same crop metadata saved with the classifier, then predicts defect type from that crop.

Ground-truth crops produced an apparently large screw gain (0.444 historical full-image accuracy → 0.861 oracle-crop accuracy), but the same model fell to 0.194 when evaluated with PatchCore-predicted crops. That is train/inference crop-domain mismatch, so the oracle result is not used for deployment. Retraining directly on PatchCore crops and seeding both alternatives at 42 gives the fair comparison:

| Screw ResNet18 input | Val accuracy | Crop available at deployment? |
|---|---:|---|
| Full image, seeded rerun | 0.4722 | Yes |
| Ground-truth-mask crop | 0.8611 | No (oracle only) |
| Ground-truth-trained model, PatchCore crop at evaluation | 0.1944 | Yes, but mismatched training domain |
| PatchCore crop for both train and validation | **0.6111** | **Yes** |

The deployment-matched crop improves screw by **13.89 accuracy points** on the identical 83/36 split. It correctly classifies all 7 `manipulated_front`, 6/7 `scratch_head`, and 7/8 `scratch_neck` validation images, while `thread_side`/`thread_top` remain difficult. Oracle focused crops did not establish a general category improvement: compared with the historical full-image runs, bottle scored 0.684 vs 0.789, hazelnut 0.810 vs 0.857, carpet 0.889 vs 0.815, and leather 0.893 vs 0.964; deployment-time predicted crops did not beat the full-image model for any of those four. Therefore only [config/screw_config.yaml](config/screw_config.yaml) enables `defect_classifier.use_focused_crops`; every other category explicitly keeps it false.

**Methodological caveat:** PatchCore's pixel threshold is selected from the labeled test set in the existing anomaly-evaluation pipeline, so this crop experiment inherits that optimistic calibration and remains a same-dataset ablation, not an independent production estimate. A stronger follow-up would calibrate the pixel threshold on a separate validation set and evaluate defect typing on untouched images.
- **Defect-focused crops only help when training and deployment use the same localization source.** Ground-truth-mask crops raised screw's oracle validation accuracy to 0.8611, but that model collapsed to 0.1944 when fed deployable PatchCore crops. Training and validating on PatchCore crops instead improved the reproducibly seeded full-image result from 0.4722 to 0.6111. The gain is enabled only for screw because predicted crops did not beat full images elsewhere, and it still inherits the detector pixel threshold's same-test-set calibration.

### Classical ML alternative: PCA + LDA defect-type classifier

[src/models/train_pca_lda_defect_classifier.py](src/models/train_pca_lda_defect_classifier.py) targets the exact same per-category defect-type problem and split as the table above, but replaces the fine-tuned CNN head with a classical pipeline: frozen ImageNet-pretrained ResNet18 embeddings (512-d, no fine-tuning) → `StandardScaler` → PCA (30 components) → Linear Discriminant Analysis as the classifier itself.

```bash
python -m src.models.train_pca_lda_defect_classifier --config config/screw_config.yaml
```

| Category | Train / val images | Deep classifier (CNN head) val accuracy | PCA+LDA val accuracy |
|---|---|---|---|
| Screw | 83 / 36 | 0.444 | **0.528** |
| Bottle | 44 / 19 | 0.789 | **0.842** |
| Hazelnut | 49 / 21 | **0.857** | 0.714 |
| Carpet | 62 / 27 | **0.815** | 0.593 |
| Leather | 64 / 28 | **0.964** | 0.750 |

**Finding — PCA+LDA helps exactly where the deep classifier was weakest, and hurts where it was already strong.** On screw and bottle — the two categories where the fine-tuned CNN head scored worst (0.444, 0.789) — PCA+LDA improved val accuracy by 8–5 points. On hazelnut/carpet/leather, where the CNN head was already doing well (0.815–0.964), PCA+LDA was clearly worse. This is consistent with *why* each method should win in each regime: with only ~16–20 train images per class, fine-tuning a full CNN head (thousands of parameters) is prone to overfitting/underfitting noise, while LDA fits far fewer parameters (a linear decision boundary per class pair in a 30-d PCA-reduced space) directly from the same frozen embeddings — a better-conditioned problem exactly when data per class is scarcest. But LDA's decision boundary is linear and the embeddings aren't fine-tuned to the category's specific defects, so once the CNN head has enough signal to learn a good nonlinear boundary (leather, carpet, hazelnut), it pulls ahead. **Lesson:** there's no universally-better model here — for a genuinely tiny, hard fine-grained problem (screw), a lower-capacity classical classifier on frozen features can beat a fine-tuned CNN head; for a moderately-sized, visually-separable one, fine-tuning wins. Try both and pick per category rather than assuming the deep model is always the right default.

## Notes & Lessons Learned

- **The baseline supervised classifier's near-perfect scores are misleading.** Since MVTec's `train/` only contains `good` images, a supervised good-vs-defective classifier has to be trained on a split carved out of `test/` — meaning every defect *type* it's evaluated on was already seen during training. This produced accuracy/precision/recall/F1 all at 1.0, which reflects a tiny, easy, non-independent validation split rather than real-world generalization. Treat this model as a baseline/sanity-check only.
- **Best-epoch restoration prevents a late training epoch from silently becoming the deployed model, while cross-validation exposes split sensitivity.** The baseline trainer now measures validation loss after every epoch, restores the lowest-loss weights, supports patience-based early stopping, and offers stratified K-fold evaluation with mean ± standard deviation and aggregate out-of-fold metrics. In the final seeded SimpleCNN verification run, validation loss improved through all 10 epochs (best/final validation loss 0.1645 at epoch 10), so early stopping correctly did not fire and accuracy/F1 remained 1.00/1.00; this confirms the control does not truncate a still-improving model. The K-fold mode saves evaluation artifacts separately and does not overwrite the deployable single-split checkpoint.
- **The unsupervised PatchCore detector is the more trustworthy signal.** Trained only on `train/good` (no labels at all) and evaluated on the *entire* labeled test set, it scored a more believable 0.91 image-level ROC-AUC and 0.97 pixel-level AUROC — a much fairer estimate of how the system would behave on truly unseen defects.
- **A stronger backbone improves detection but not localization tightness.** Swapping PatchCore's frozen feature extractor from ResNet18 to WideResNet50-2 raised image ROC-AUC from 0.909 to 0.935, but mean IoU/Dice stayed roughly the same — both backbones produce patch features at the same coarse 28×28 grid, so the bilinear-upsampling blur (not backbone capacity) is the bottleneck for tight segmentation.
- **PatchCore capacity and feature depth interact rather than improving monotonically.** On screw, halving the memory bank to 1,000 patches reduced image AUROC to 0.8664 and `layer3` alone reduced it to 0.8000, while independently increasing the bank to 4,000, widening the projection to 256, or retaining only `layer2` each helped. Combining those three settings reached 0.9696 ± 0.0061 image AUROC and improved IoU/Dice to 0.0677/0.1237 across three seeds. The gain costs about 2× memory-bank storage and scoring work, and remains category-specific; tune capacity, representation, and compute together instead of assuming "larger" or "deeper" is universally better.
- **k-NN patch scoring can smooth single-neighbor noise, but it is an objective-dependent tradeoff rather than an automatic upgrade.** On the default screw detector, k=3 raised image/pixel AUROC from 0.9092/0.9761 to 0.9389/0.9783 and improved IoU/Dice; it also helped hazelnut and leather localization. On the already-tuned screw detector, however, three seeds showed higher pixel AUROC but lower image AUROC and slightly lower overlap. Softmax image-score reweighting did not improve any current non-ceiling category. Keep both controls explicit and category-tested rather than stacking every individually plausible PatchCore option.
- **A promising single-seed gain can shrink or reverse under multi-seed, multi-category testing.** PCA coreset projection initially appeared to match the WideResNet50-2 upgrade on screw at seed 42 (image AUROC 0.9092→0.9356), but three seeds reduced the PCA mean to 0.9212 ± 0.0144; one seed was below the random baseline, and mean screw IoU/Dice also decreased slightly. Across bottle/hazelnut/carpet/leather, image gains were at most 0.15 points and pixel AUROC was effectively unchanged. See [PatchCore coreset projection: random (JL) vs PCA](#patchcore-coreset-projection-random-jl-vs-pca--an-ablation). **Lesson:** report distributions rather than promoting the best seed, test optimization ideas across categories, and keep them config-driven until the evidence supports a new default.
- **An ensemble can't out-perform a classifier that's already at its evaluation ceiling.** Fusing the supervised classifier with PatchCore's score seemed like an obvious way to get the best of both, but on the classifier's own held-out validation subset the classifier alone already hits AUROC 1.0 (for the same reason its plain validation metrics are inflated — see above), so there's no room left for the ensemble to improve on. A fusion technique can only be shown to add value when tested on a sample that's genuinely novel to *every* component model, not just to one of them.
- **High pixel AUROC does not imply tight defect segmentation.** Mean IoU/Dice came out low (~0.04/0.07) even though pixel AUROC was high. The anomaly map is produced on a coarse 28×28 feature grid and bilinearly upsampled to 224×224, so it's good at *ranking* defect pixels highly (localizing the general region) but blurry compared to MVTec's tight ground-truth masks. Pixel AUROC and IoU/Dice answer different questions and should be reported together, not interchangeably.
- **`.gitignore` patterns without a leading `/` match at any depth.** An earlier `data/` rule (meant to exclude the raw MVTec dataset) was silently also excluding `src/data/`, so real source code was never staged. Always sanity-check ignore rules with `git check-ignore -v <path>` and `git status --ignored` before trusting `git add -A`.
- **Virtual environments need explicit, exact ignore entries.** `.venv/` in `.gitignore` did not match a second environment folder named `.venv-1/`, which had all project dependencies installed — a `git add -A` would have swept hundreds of MB of installed packages into the repo if left unnoticed.
- **Without foreground masking, the anomaly heatmap bleeds into the background.** Since PatchCore scores every patch (including plain background), and each image's heatmap is min-max normalized independently, small background texture/contrast differences got stretched into visible red/yellow — mimicking a real defect signal even though nothing was wrong there. Adding a simple Otsu-threshold foreground mask ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py)) to suppress background patches (pinning them to the object's minimum anomaly distance) fixed this without hurting detection accuracy (image AUROC 0.91 → 0.909, pixel AUROC 0.973 → 0.976) — a good reminder that per-image score normalization can amplify noise anywhere the model doesn't explicitly ignore it.
- **A plain min-max colored heatmap doesn't visually match the actual decision boundary.** Even with background masked out, ordinary (sub-threshold) screw texture — like normal thread ridges — still has a non-zero, spatially-varying anomaly score, so a full-range jet colormap can render it yellow/green and look like "a big defect" even when the real decision (score vs. the calibrated threshold) says otherwise. Anchoring the colormap to the decision threshold (`normalize_map_threshold` in [src/visualization/heatmap.py](src/visualization/heatmap.py)) — compressing below-threshold values into the cool half and only letting above-threshold values read as hot — makes the heatmap visually agree with the Normal/Defective/severity verdict.
- **`matplotlib.imshow` silently re-normalizes data unless you pass `vmin`/`vmax`.** After pre-compressing anomaly values into a threshold-anchored `[0, 1]` range, `axes.imshow(normalized, cmap="jet")` was auto-rescaling that already-compressed range back to the full colormap, quietly undoing the fix for one figure panel. Any time you pass pre-normalized data to `imshow`, pass `vmin=0, vmax=1` explicitly or matplotlib will stretch contrast based on the data's own min/max.
- **A vision heuristic tuned on one category can silently break another.** The foreground-masking fix that helped `screw` (and generalized fine to `hazelnut`) made `bottle`'s pixel-level localization *worse than random* until it was made a per-category, config-driven toggle instead of an always-on assumption — see [Generalization to Other Categories](#generalization-to-other-categories) above. Multi-category testing caught this; single-category testing would not have.
- **Recognizing the object type is a much easier task than recognizing its defects.** The category classifier ([src/models/train_category_classifier.py](src/models/train_category_classifier.py)) hits 100% validation accuracy telling all eight categories (screw, bottle, hazelnut, carpet, leather, wood, grid, tile) apart, in contrast to the earlier "misleadingly perfect" good-vs-defective classifier finding. This is expected and not a red flag the same way: whole object types differ enormously in shape/texture/color (an easy, well-separated classification problem), while a defect is a subtle local deviation within one object type (a hard, fine-grained problem) — a perfect score means something very different depending on which of the two problems is being solved.
- **A lesson learned from one category, once turned into a config option, actually transfers.** Adding `carpet` and then `leather` (both full-frame textures like `bottle`) with `use_foreground_mask: false` set from the start — instead of rediscovering the problem — produced the best pixel-level ROC-AUC so far each time (carpet 0.987, then leather 0.991). Turning a bug fix into an explicit, per-category config decision (rather than just patching the one case that broke) is what makes a lesson actually reusable on the next category. `wood`, `grid`, and `tile` (also full-frame textures) inherited the same `use_foreground_mask: false` default unchanged when they were added later — their pixel ROC-AUC (0.930, 0.947, 0.941) confirms the toggle was still the right call for each, even though none matched leather's peak, showing the config-driven decision holds without needing to be re-derived per category even as absolute performance varies.
- **A from-scratch CNN needs far more training than a pretrained backbone on a small dataset.** Adding a `simple_cnn` option (small sequential CNN, no pretrained weights) to [src/models/baseline_classifier.py](src/models/baseline_classifier.py) and training it side-by-side with ResNet18/EfficientNet-B0 on the same 112 images/10 epochs showed all three reach the same (misleadingly perfect) validation score, but their training loss tells a very different story: ResNet18/EfficientNet-B0 converge to ~0.01–0.04 while the from-scratch CNN is still at ~0.17. When validation metrics saturate/tie across models (often a sign the eval set is too small or too easy), check the training loss curve too — it can reveal a real gap that accuracy alone hides.
- **A larger or newer architecture is not automatically a performance improvement on tiny data.** Adding ImageNet-pretrained ConvNeXt-Tiny and ViT-B/16 to the same screw split/10-epoch comparison produced the same 1.00 validation accuracy/F1 as every existing supervised model. ConvNeXt converged most strongly (final loss 0.0024), but its 27.8M parameters did not beat the 4.0M EfficientNet-B0 on validation metrics; ViT-B/16 was much larger (85.8M), less stable during training (epoch-4 loss spike to 1.057), and ended at 0.081 loss without improving validation performance. On 112 labeled training images, transfer learning makes both usable, but the CNN inductive bias and smaller footprint remain a better fit than a large transformer. Architecture comparisons need independent data or cross-validation once a tiny held-out split saturates.
- **`torch.cuda.is_available()` alone misses Apple Silicon GPUs.** The training scripts ([train_baseline.py](src/models/train_baseline.py), [train_category_classifier.py](src/models/train_category_classifier.py), [run_anomaly_detection.py](src/models/run_anomaly_detection.py)) only checked for CUDA and silently fell back to CPU on this Mac, even though `torch.backends.mps.is_available()` was `True`. Adding an explicit `cuda` → `mps` → `cpu` fallback let the category classifier retrain (1631 images, 8 epochs) run on the Apple Silicon GPU instead of CPU with no code/behavior change beyond speed. Always check for `mps` explicitly on Apple Silicon — `cuda.is_available()` being `False` doesn't mean no GPU is available.
- **A fine-grained per-category defect-type classifier is a fundamentally harder problem than the coarse good/defective or category classifiers, and doesn't automatically inherit their near-perfect scores — scaling from five to eight categories reinforced this rather than changing it.** See [Defect-Type Classification](#defect-type-classification-per-category) above: screw scored only 0.444 validation accuracy across its 5 defect types (~16–17 train images/class, and the defect types are subtle geometric variations of each other) while leather scored 0.964 with a similar amount of data but visually distinct defect types. Adding `wood`, `grid`, and `tile` only widened the spread: **grid** scored even worse than screw (0.222 — its bent/broken/glue/metal_contamination/thread defects are subtle spatial/material variations on a regular pattern, with `glue` absorbing most misclassifications), **tile** matched leather's success (0.923 — crack/glue_strip/gray_stroke/oil/rough are visually distinct surface phenomena), and `wood` landed in between (0.667 — some distinct defect types like `hole`/`scratch`, but `combined` and `color` are hard to separate from natural grain variation). Class count and dataset size alone still don't predict fine-grained accuracy across all eight categories — inter-class visual similarity is the dominant factor, and it must be checked per category rather than assumed to generalize from any prior category's result.
- **A high anomaly score on an unfamiliar photo is domain shift, not overfitting — and the two need different evidence.** Testing the hazelnut PatchCore detector on real held-out MVTec test images it never trained on cleanly separates good (scores 2.20–2.39) from defective (2.88–3.45) around its 2.618 threshold, confirming it generalizes fine *within its training distribution* (consistent with the 0.998 image ROC-AUC reported above). Feeding it two web images instead — a glossy stock-photo hazelnut on a transparent background (score 3.68, "Defective") and an unrelated peach that the category classifier only weakly matched to hazelnut at 60% confidence (score 4.05, "Defective") — both scored *above* every real defective test image. PatchCore is a memory-bank method: it has no notion of "same object, different defect" vs. "different lighting/background/object entirely" — both simply read as "far from anything memorized." The fix isn't more training, it's recognizing that image-space generalization (unseen defects on the *same* capture setup) and domain generalization (unseen capture conditions or object types) are different questions, and this project's ROC-AUC numbers only speak to the former.
- **A lower-capacity classical classifier can beat a fine-tuned CNN head exactly when training data per class is scarcest, and lose once it isn't — and this held on two independent classifiers, not just one.** Swapping the per-category defect-type classifier's fine-tuned CNN head for a classical PCA (30 components) + LDA classifier on frozen ResNet18 embeddings ([train_pca_lda_defect_classifier.py](src/models/train_pca_lda_defect_classifier.py)) improved val accuracy on the two hardest categories (screw 0.444→0.528, bottle 0.789→0.842, ~16–20 train images/class) but made it worse on the three easier ones (hazelnut, carpet, leather, all already 0.815–0.964) — see [Classical ML alternative](#classical-ml-alternative-pca--lda-defect-type-classifier) above. Applying the *identical* PCA+LDA recipe to the coarser good-vs-defective baseline classifier ([train_pca_lda_baseline_classifier.py](src/models/train_pca_lda_baseline_classifier.py)) reproduced the same pattern independently: it only beat the deep classifier on bottle (0.88→0.96 accuracy), the one category where the deep classifier itself was weakest, and lost everywhere else (see [Classical ML alternative: PCA + LDA baseline classifier](#classical-ml-alternative-pca--lda-baseline-classifier)). Fewer fitted parameters (a linear boundary in a 30-d PCA space vs. a full CNN head) is a better-conditioned fit when samples/class are scarce, but caps accuracy once there's enough data/separability for the CNN's nonlinear boundary to pay off — seeing this repeat on two different classifiers/problems is good evidence it's a general property of the data regime, not a one-off fluke of one dataset split. **Also:** on this Mac's Apple Silicon `numpy` (Accelerate BLAS backend), PCA's internal matmul emitted spurious `divide by zero`/`overflow` `RuntimeWarning`s on these small (~80×512) matrices even though the actual transformed values were always finite/correct (verified by direct inspection) — a known Accelerate quirk, not a real numerical bug; safe to suppress rather than chase.
- **`train_category_classifier.py` and `train_defect_classifier.py` originally didn't seed PyTorch, so their reported accuracy was one sample from a distribution, not a fixed number.** Unlike `train_baseline.py` (`torch.manual_seed(data_cfg["seed"])` before every run), these two scripts only seeded the train/val split, not model init/training — confirmed by re-running transistor's defect-type classifier back-to-back with identical configs: 0.750, then 0.583, then 0.500 val accuracy (on a 12-image val set, each image is ±8.3 points). Adding per-class `classification_report` output (precision/recall/F1/support per class, not just aggregate accuracy + a raw confusion matrix) to both scripts made this variance visible instead of hiding behind one aggregate number. Both scripts are now seeded like `train_baseline.py` (fixed in the same change that added early stopping + best-epoch restoration to them), and the seeded transistor run reports 0.750.
- **Overfitting controls are not interchangeable — on the same tiny dataset, best-epoch restoration helped while backbone freezing badly hurt.** The transistor defect-type training curves showed classic overfitting (train accuracy ~0.93–0.96 vs. val stuck at 0.25–0.50, val loss plateauing while train loss kept falling), so three controls were added to [train_defect_classifier.py](src/models/train_defect_classifier.py): seeding, early stopping with best-epoch restoration (borrowed from `train_baseline.py`), and a `--freeze-backbone` option (~2k trainable head parameters instead of ~11M). Seeded fine-tuning with best-epoch restoration scored 0.750 val accuracy (val loss still improving at epoch 10, so early stopping correctly didn't fire). Freezing the backbone scored 0.167 at the config's fine-tuning LR (1e-4; the head barely moved and early-stopped at epoch 4) and only 0.417 even at a head-appropriate LR of 1e-2 — roughly half the fine-tuned result. This mirrors the PCA+LDA lesson from the opposite direction: frozen ImageNet features are only competitive when fine-tuning itself is failing; transistor's defect types (bent/cut leads, misplaced parts) are apparently distinguishable enough for fine-tuning to win, so capacity reduction just threw away useful adaptation. Try controls one at a time and measure — "reduces overfitting" doesn't mean "improves validation accuracy."
- **When validation loss is still falling at the epoch limit, raising the budget confirms whether the model has actually converged.** Re-running the same seeded transistor fine-tune with a `--epochs 30` override (added alongside `--learning-rate` for exactly this check) showed val loss keep falling two more epochs (0.885→0.885, effectively flat) before turning back up (0.912, 0.922, 0.969), triggering early stopping at epoch 13 with the *same* best epoch (10) and *same* val accuracy (0.750) as the original 10-epoch run. That confirms epoch 10 was the true optimum, not an artifact of an undersized epoch budget — the extra epochs bought no additional accuracy, only a cleaner, textbook-shaped loss curve (rise, trough, early-stop) to point at as evidence.
