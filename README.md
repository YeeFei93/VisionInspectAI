# VisionInspectAI
Use MVTec-AD categories to detect whether an image is normal or defective, and show the defect location using a heatmap. Currently trained/evaluated end-to-end on `screw`, `bottle`, `hazelnut`, `carpet`, and `leather`; the Streamlit demo auto-detects which one was uploaded.

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

At minimum, grab the categories this project is already configured for: `screw`, `bottle`, `hazelnut`, `carpet`, `leather`. (You only need the top-level dataset archive, or the individual per-category archives for just those five.)

### 3. Build manifests, train the models, and launch the demo

Repeat for each category (`screw`, `bottle`, `hazelnut`, `carpet`, `leather`):

```bash
python -m src.data.create_manifest --category screw
python -m src.models.train_baseline --config config/screw_config.yaml
python -m src.models.run_anomaly_detection --config config/screw_config.yaml
```

Then train the category classifier (needed for the Streamlit demo's auto-detect feature) once all five manifests exist:

```bash
python -m src.models.train_category_classifier --categories screw bottle hazelnut carpet leather
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
  - Supervised: [src/models/baseline_classifier.py](src/models/baseline_classifier.py) — a `resnet18` / `efficientnet_b0` (transfer learning) or `simple_cnn` (small CNN trained from scratch) image classifier fine-tuned on labeled good/defective images (trained via [src/models/train_baseline.py](src/models/train_baseline.py)).
  - Unsupervised: [src/models/anomaly_detector.py](src/models/anomaly_detector.py) — a PatchCore-style anomaly detector trained only on `train/good` images (no defect labels needed), evaluated on the full labeled test set via [src/models/run_anomaly_detection.py](src/models/run_anomaly_detection.py).

- **Machine learning / deep learning techniques**
  - Deep CNN backbones (ResNet18) pretrained on ImageNet, used both for end-to-end supervised fine-tuning and as a frozen feature extractor.
  - Classical ML techniques layered on top of deep features: greedy k-center coreset subsampling and nearest-neighbor distance scoring for anomaly detection ([src/models/anomaly_detector.py](src/models/anomaly_detector.py)), plus standard evaluation metrics (accuracy/precision/recall/F1, ROC-AUC, IoU, Dice) in [src/evaluation/metrics.py](src/evaluation/metrics.py).

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
- `src/evaluation/` — classification and pixel-level localization metrics
- `src/visualization/` — anomaly heatmap rendering
- `notebooks/` — dataset exploration
- `app/` — Streamlit inspection demo
- `config/` — per-category YAML configs (e.g. `screw_config.yaml`)
- `tests/` — unit tests for the pure evaluation/preprocessing/visualization functions

## Quick Start

```bash
# 1. Build the manifest CSV for a category
python -m src.data.create_manifest --category screw

# 2. Train the supervised baseline classifier
python -m src.models.train_baseline --config config/screw_config.yaml

# 3. Train and evaluate the unsupervised PatchCore anomaly detector
python -m src.models.run_anomaly_detection --config config/screw_config.yaml

# 4. (optional) Train the per-category defect-type classifier (what kind of defect is it?)
python -m src.models.train_defect_classifier --config config/screw_config.yaml

# 5. Launch the interactive demo (auto-detects the category from the uploaded image)
streamlit run app/streamlit_app.py

# 6. Run the unit tests
python -m pytest tests/ -v
```

To compare baseline classifier architectures, point `train_baseline.py` at any of `config/screw_config.yaml` (ResNet18), `config/screw_config_efficientnet_b0.yaml`, or `config/screw_config_simple_cnn.yaml` — each writes its own checkpoint/metrics file so results don't overwrite each other.

To run the full pipeline on a different MVTec-AD category, copy `config/screw_config.yaml` to `config/<category>_config.yaml`, update `category` and `data.manifest_path`, then repeat steps 1–4 with `--category <category>` / `--config config/<category>_config.yaml`. Already set up this way: `screw`, `bottle`, `hazelnut`, `carpet`, `leather` (see [Generalization to Other Categories](#generalization-to-other-categories) below). After adding a new category, retrain the category classifier so the Streamlit demo can auto-detect it too:

```bash
python -m src.models.train_category_classifier --categories screw bottle hazelnut carpet leather <new_category>
```

## Pipeline Steps in Detail

### Step 1 — Build the manifest CSV

**Script:** [src/data/create_manifest.py](src/data/create_manifest.py)

**What it does:** walks `data/mvtec_anomaly_detection/<category>/train/good/` (all "good" images) and every `test/<defect_type>/` subfolder (`good` plus each defect type, e.g. `manipulated_front`, `scratch_head`), pairing each defective test image with its ground-truth mask from `ground_truth/<defect_type>/<name>_mask.png` when one exists. This is bookkeeping only — no model/algorithm involved, just a filesystem scan + CSV writer.

**Output:** `data/manifests/<category>.csv` with columns `image_path, mask_path, split, label, defect_type` (`label` = 0 good / 1 defective). This manifest is the single source of truth every downstream script reads from — e.g. [data/manifests/screw.csv](data/manifests/screw.csv).

### Step 2 — Train the supervised baseline classifier

**Script:** [src/models/train_baseline.py](src/models/train_baseline.py), using [src/data/dataset.py](src/data/dataset.py) (`ManifestImageDataset`, `make_train_val_split`), [src/preprocessing/transform.py](src/preprocessing/transform.py) (resize/normalize + light augmentation for train, deterministic resize/normalize for val) and [src/models/baseline_classifier.py](src/models/baseline_classifier.py) (`build_baseline_model`).

**What it does:** since MVTec's `train/` only has `good` images, this baseline instead carves a train/val split out of the *labeled* `test/` rows (`data.val_split` / `data.seed` in the config, e.g. 70/30 for screw) and trains a plain good-vs-defective image classifier on it.

**Algorithm:** one of three interchangeable architectures (`model.architecture` in the config):
- `resnet18` or `efficientnet_b0` — ImageNet-pretrained CNN backbone (transfer learning) with the final layer replaced by a 2-way linear head.
- `simple_cnn` — a small 4-block Conv→BatchNorm→ReLU→MaxPool CNN ([SimpleCNN](src/models/baseline_classifier.py)) trained from scratch (no pretrained weights), used as a from-scratch comparison point.

Trained with cross-entropy loss and the Adam optimizer for `train.epochs` epochs (`train.learning_rate`, `train.batch_size` from the config).

**Output** (named `baseline_<architecture>_<category>`, so different architectures don't overwrite each other):
- `models/checkpoints/baseline_<architecture>_<category>.pt` — model state dict.
- `outputs/metrics/baseline_<architecture>_<category>_metrics.json` — accuracy, precision, recall, F1, confusion matrix on the held-out val subset.
- `outputs/figures/baseline_<architecture>_<category>_confusion_matrix.png` — plotted confusion matrix.

### Step 3 — Train and evaluate the unsupervised PatchCore anomaly detector

**Script:** [src/models/run_anomaly_detection.py](src/models/run_anomaly_detection.py), using [src/models/anomaly_detector.py](src/models/anomaly_detector.py) (`PatchCoreAnomalyDetector`), [src/preprocessing/segmentation.py](src/preprocessing/segmentation.py) (foreground mask), [src/evaluation/metrics.py](src/evaluation/metrics.py) and [src/visualization/heatmap.py](src/visualization/heatmap.py).

**What it does:** fits the detector on `train/good` only (no labels used at all), then scores every image in the full labeled `test/` split (good + every defect type).

**Algorithm — PatchCore** (Roth et al., *"Towards Total Recall in Industrial Anomaly Detection"*, CVPR 2022; this implementation keeps the core recipe but drops the paper's optional softmax re-weighting term):
1. **Locally-aware patch features:** a frozen, ImageNet-pretrained backbone (`anomaly_detection.backbone`: `resnet18` or `wide_resnet50_2`) extracts feature maps from two intermediate layers (`layer2`, `layer3`), each 3×3-average-pooled for local context and concatenated into one multi-scale patch feature map.
2. **Memory bank via greedy coreset:** all patch features from every `train/good` image are subsampled with a greedy k-center coreset algorithm (`_greedy_coreset`) — starting from a random patch, it repeatedly keeps the patch farthest (in a Johnson–Lindenstrauss-projected space, for speed) from everything already selected — down to `anomaly_detection.coreset_ratio` of the pool, capped at `max_coreset_size` (2000) patches. This keeps the bank small while preserving diversity.
3. **Scoring:** for a query image, each patch's anomaly score is its distance to the nearest neighbor in the memory bank; the image-level score is the max over patches (optionally after `compute_foreground_mask` pins background patches to the object's minimum distance, so only the object surface can drive the score); the per-patch score grid is bilinearly upsampled to the input resolution to form the anomaly heatmap.
4. **Thresholding:** the good/defective decision boundary is chosen by maximizing Youden's J statistic (`youden_threshold`) on the ROC curve of image scores over the test set.

**Evaluation** ([src/evaluation/metrics.py](src/evaluation/metrics.py)): image-level ROC-AUC + accuracy/precision/recall/F1 at the Youden threshold, plus pixel-level ROC-AUC (heatmap vs. ground-truth mask over every test pixel) and mean IoU/Dice (predicted vs. ground-truth defect mask at the pixel-level Youden threshold, defect images only).

**Output** (named `patchcore_<backbone>_<category>`):
- `models/checkpoints/patchcore_<backbone>_<category>_memory_bank.pt` — the memory bank tensor (the "trained model").
- `outputs/metrics/patchcore_<backbone>_<category>_metrics.json` — AUROC, threshold, score min/max, pixel AUROC, pixel threshold, mean IoU, mean Dice, plus accuracy/precision/recall/F1.
- `outputs/heatmaps/patchcore_<backbone>_<category>_<defect_type>_example.png` — one example figure per defect type: original image | predicted heatmap | ground-truth mask | overlay, for a quick visual sanity check.

### Step 4 — Train the category classifier (auto-detect object type)

**Script:** [src/models/train_category_classifier.py](src/models/train_category_classifier.py).

**What it does:** combines every requested category's manifest (train + test rows, good and defective alike — object-type recognition doesn't care about defect status), replaces the good/defective label with a category index, and trains a multi-class "which object is this?" classifier on a stratified train/val split.

**Algorithm:** ImageNet-pretrained ResNet18 (`build_baseline_model`) with an `N`-way head (`N` = number of categories), trained with cross-entropy + Adam for 8 epochs by default.

**Output:**
- `models/checkpoints/category_classifier_resnet18.pt` — model state dict.
- `outputs/metrics/category_classifier_metrics.json` — the ordered category list (defines the label→name mapping), validation accuracy, and confusion matrix.

### Step 5 — Train the defect-type classifier (per category)

**Script:** [src/models/train_defect_classifier.py](src/models/train_defect_classifier.py).

**What it does:** unlike the Step 2 good/defective classifier, this only looks at a single category's *defective* test/ rows and predicts which `defect_type` it is (e.g. leather: `color`/`cut`/`fold`/`glue`/`poke`), on a stratified train/val split over just those defect types.

**Algorithm:** the same `build_baseline_model` architecture as the category's `config/<category>_config.yaml` (ResNet18 by default), with an `N`-way head (`N` = number of defect types for that category), trained with cross-entropy + Adam for 10 epochs.

**Output** (named `defect_classifier_<architecture>_<category>`):
- `models/checkpoints/defect_classifier_<architecture>_<category>.pt` — model state dict.
- `outputs/metrics/defect_classifier_<architecture>_<category>_metrics.json` — the ordered defect-type list (defines the label→name mapping), validation accuracy, and confusion matrix.

This is optional per category — the Streamlit demo checks whether a checkpoint/metrics file exists for the detected category and simply skips the defect-type display (with a hint to train it) if not.

### Step 6 — Launch the interactive Streamlit demo

**Script:** [app/streamlit_app.py](app/streamlit_app.py).

**What it does, end to end, for an uploaded image:**
1. Runs the Step 4 category classifier to auto-detect the object type (`screw`/`bottle`/`hazelnut`/`carpet`/`leather`), with a collapsed manual-override dropdown as a fallback.
2. Loads that category's config, PatchCore memory bank (Step 3 checkpoint) and metrics file (for the decision threshold).
3. Computes the Otsu foreground mask (`compute_foreground_mask`) unless the category's config sets `use_foreground_mask: false` (full-frame textures like `bottle`/`carpet`/`leather`).
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

**What's tested:** the pure, model-free functions so correctness doesn't depend on having trained anything first —
- [tests/test_metrics.py](tests/test_metrics.py): `compute_classification_metrics`, `youden_threshold`, `compute_iou`/`compute_dice`, `compute_pixel_level_metrics` against hand-built synthetic labels/scores/masks.
- [tests/test_heatmap.py](tests/test_heatmap.py): `normalize_map`/`normalize_map_threshold`/`make_overlay` produce correctly-shaped, correctly-ranged outputs.
- [tests/test_segmentation.py](tests/test_segmentation.py): `compute_foreground_mask` correctly separates a synthetic bright object from a dark background (and vice versa).

**Output:** pytest pass/fail report in the terminal; no files are written.

## Model Comparison (Screw Category)

Same train/val split (112/48 images carved from `test/`), same 10 epochs, same optimizer/learning rate — only the architecture changes.

| Model | Type | Params | Pretrained | Final train loss | Val accuracy / precision / recall / F1 |
|---|---|---|---|---|---|
| ResNet18 | Transfer learning (CNN) | 11.2M | ✅ ImageNet | 0.010 | 1.00 / 1.00 / 1.00 / 1.00 |
| EfficientNet-B0 | Transfer learning (CNN) | 4.0M | ✅ ImageNet | 0.041 | 1.00 / 1.00 / 1.00 / 1.00 |
| Simple CNN (sequential) | From-scratch CNN | 0.25M | ❌ | 0.175 | 1.00 / 1.00 / 1.00 / 1.00 |
| PatchCore (ResNet18 features) | Unsupervised anomaly detection | — | ✅ ImageNet (frozen) | n/a | Image ROC-AUC 0.91, Pixel ROC-AUC 0.98 |

**Finding:** all three supervised classifiers hit the same perfect validation score, because (per the lesson below) the validation split is tiny and every defect type in it was already seen during training — accuracy/F1 can't distinguish them. The **training loss curve is the more informative signal**: the two pretrained/transfer-learning models (ResNet18, EfficientNet-B0) converge to a near-zero loss within 10 epochs, while the from-scratch Simple CNN — with ~45x fewer parameters and no ImageNet prior — is still at 0.175 and visibly still improving. This is the expected, textbook result for a ~300-image dataset: transfer learning from a pretrained backbone reaches a good fit far faster than training a CNN from scratch, which would need many more epochs (and/or more data or augmentation) to close the gap. **Recommendation:** for this dataset size, prefer a pretrained backbone (ResNet18 or the smaller/cheaper EfficientNet-B0) over a from-scratch CNN; use PatchCore (unsupervised) as the primary, more trustworthy detector since it doesn't depend on this leaky supervised validation split at all.

### PatchCore backbone: ResNet18 vs WideResNet50-2

Same memory bank size (2000 patches), same coreset/foreground-masking settings — only the frozen feature-extractor backbone changes (`config/screw_config_wide_resnet50_2.yaml`).

| Backbone | Image ROC-AUC | Pixel ROC-AUC | Mean IoU | Mean Dice |
|---|---|---|---|---|
| ResNet18 | 0.909 | 0.976 | 0.047 | 0.088 |
| WideResNet50-2 | **0.935** | **0.978** | 0.044 | 0.083 |

**Finding:** the larger WideResNet50-2 backbone gives richer patch features, improving image-level ROC-AUC by ~2.6 points (0.909 → 0.935) and pixel-level ROC-AUC slightly, at the cost of a much larger download/forward pass (~264MB vs ~45MB, noticeably slower per image on CPU). Mean IoU/Dice don't improve — both backbones extract features at the same 28×28 spatial grid, so the blur from bilinear upsampling (the actual limiting factor for tight segmentation, per the lesson below) is unaffected by backbone size. **Recommendation:** use WideResNet50-2 when detection accuracy matters more than latency/footprint (e.g. batch QA review); keep ResNet18 for fast/interactive use (e.g. the Streamlit demo).

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

The same pipeline (manifest → baseline classifier → PatchCore → Streamlit demo) was run end-to-end on four more MVTec-AD categories, picked to be different in shape: `bottle` (top-down shot of a bottle mouth), `hazelnut` (small object on a plain background, closer to `screw`), `carpet` (a close-up textile texture filling the whole frame, no discrete object at all), and `leather` (another close-up, full-frame texture, same situation as carpet).

| Category | Classifier val accuracy/F1 | PatchCore image ROC-AUC | PatchCore pixel ROC-AUC | Mean IoU | Mean Dice |
|---|---|---|---|---|---|
| Screw | 1.00 / 1.00 | 0.909 | 0.976 | 0.047 | 0.088 |
| Bottle | 0.88 / 0.92 | 1.000 | 0.981 | 0.397 | 0.550 |
| Hazelnut | 1.00 / 1.00 | 0.998 | 0.970 | 0.209 | 0.320 |
| Carpet | 0.92 / 0.94 | 0.971 | 0.987 | 0.251 | 0.372 |
| Leather | 1.00 / 1.00 | 1.000 | **0.991** | 0.127 | 0.210 |

**Finding — the classifier's "perfect scores" issue isn't universal.** Unlike screw and hazelnut, the bottle classifier scored a believable 0.88 accuracy / 0.92 F1, not 1.0 (its held-out val subset is smaller — only 25 images — and the defects are more subtle). This is a useful counter-example confirming that the earlier "misleadingly perfect" finding is specifically a symptom of the *screw* dataset being small/easy, not a bug in the evaluation code.

**Finding — the foreground-masking heuristic doesn't generalize automatically, and blindly applying it can actively hurt localization.** The Otsu-based foreground mask ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py)) assumes a plain background with the object as the minority of pixels — true for screw and hazelnut, but **false for bottle**, whose images are a top-down shot where the bottle mouth fills the entire frame. Applying it anyway made bottle's pixel-level ROC-AUC **worse than random (0.374)**: Otsu split the frame into the dark inner bottle opening vs. the lighter rim, and incorrectly zeroed out real defect pixels that happened to fall inside the dark "background" region. Adding a `use_foreground_mask: false` toggle to `bottle_config.yaml` (and threading it through [run_anomaly_detection.py](src/models/run_anomaly_detection.py) and the Streamlit app) fixed it immediately: pixel ROC-AUC jumped to 0.981 and mean IoU/Dice became the *best* of screw/bottle/hazelnut (0.40 / 0.55) — confirmed visually, the predicted heatmap now matches the crescent-shaped ground-truth defect almost exactly. **Lesson:** any hand-crafted heuristic derived from one category's visual layout should be treated as a per-category, config-driven option, not a hardcoded assumption — and always sanity-check pixel-level metrics per category rather than assuming an improvement that helped one category will help (or even be neutral for) another.

**Applying the lesson upfront — `carpet` and `leather`.** Both are full-frame textures just like `bottle` (no discrete object vs. background), so their configs were created with `use_foreground_mask: false` from the start instead of discovering the problem the hard way again. Result: carpet got the best pixel-level ROC-AUC of the first four categories (0.987), and leather pushed that further to **0.991** — the best of all five categories — on the first run each time, concrete evidence that the earlier fix generalized into a repeatable, config-driven decision rather than a one-off patch. Leather's mean IoU/Dice (0.127 / 0.210) are lower than carpet's, though — a reminder that pixel ROC-AUC (ranking) and IoU/Dice (tight overlap) are still independent axes even within the same `use_foreground_mask: false` texture group (see the IoU/Dice lesson below).

The Streamlit demo ([app/streamlit_app.py](app/streamlit_app.py)) doesn't require the user to pick a category at all: [src/models/train_category_classifier.py](src/models/train_category_classifier.py) trains a small ResNet18 classifier to recognize the object type itself (all five categories are visually distinct enough that it hits 100% validation accuracy, even with carpet and leather added), and the app runs it first on the uploaded image to auto-detect the category, then routes to that category's PatchCore detector automatically — with a collapsed "override" dropdown as a manual fallback if it's ever wrong. Adding a new category only requires a new `config/<category>_config.yaml` entry in `CATEGORY_CONFIGS` plus retraining the category classifier with it included.

## Defect-Type Classification (per category)

Beyond the binary Normal/Defective verdict, [src/models/train_defect_classifier.py](src/models/train_defect_classifier.py) trains a per-category, multi-class classifier over each category's own `defect_type` labels (defective images only, see [Step 5](#step-5--train-the-defect-type-classifier-per-category)), and [app/streamlit_app.py](app/streamlit_app.py) shows the predicted defect type + confidence whenever a "Defective" verdict is reached and a matching checkpoint exists.

| Category | Defect types | Train / val images | Val accuracy |
|---|---|---|---|
| Screw | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top (5) | 83 / 36 | 0.444 |
| Bottle | broken_large, broken_small, contamination (3) | 44 / 19 | 0.789 |
| Hazelnut | crack, cut, hole, print (4) | 49 / 21 | 0.857 |
| Carpet | color, cut, hole, metal_contamination, thread (5) | 62 / 27 | 0.815 |
| Leather | color, cut, fold, glue, poke (5) | 64 / 28 | **0.964** |

**Finding — fine-grained defect-type accuracy tracks per-class sample count, not just class count.** Screw and leather both have 5 defect types, yet screw's val accuracy (0.444) is by far the worst of the five categories while leather's (0.964) is the best. Screw's confusion matrix shows `thread_side` and `thread_top` absorbing most of the misclassifications from the other three classes (`[[2,0,0,5,0],[0,5,0,2,0],[0,1,1,5,1],[0,0,0,6,1],[0,0,0,5,2]]`) — with only 83 train images spread over 5 classes (~16–17 images/class), and screw's defect types being subtle, visually-similar deviations on the same small grey object (a scratch on the head vs. the neck, thread wear on one side vs. the top), there's neither enough data nor enough visual separation for the model to tell them apart reliably. Leather's defect types, by contrast, are visually distinct surface phenomena (a color blotch vs. a cut vs. a glue smear) despite a similarly small dataset (64 train images), so it reaches near-perfect accuracy. **Lesson:** unlike the good/defective and category classifiers, a fine-grained defect-type classifier's accuracy depends heavily on how visually distinguishable that category's specific defect types are from each other, not just on how many classes or how many total images there are — screw is a case where more data alone likely wouldn't fully fix it without also addressing the inherent visual similarity between its defect types.

## Notes & Lessons Learned

- **The baseline supervised classifier's near-perfect scores are misleading.** Since MVTec's `train/` only contains `good` images, a supervised good-vs-defective classifier has to be trained on a split carved out of `test/` — meaning every defect *type* it's evaluated on was already seen during training. This produced accuracy/precision/recall/F1 all at 1.0, which reflects a tiny, easy, non-independent validation split rather than real-world generalization. Treat this model as a baseline/sanity-check only.
- **The unsupervised PatchCore detector is the more trustworthy signal.** Trained only on `train/good` (no labels at all) and evaluated on the *entire* labeled test set, it scored a more believable 0.91 image-level ROC-AUC and 0.97 pixel-level AUROC — a much fairer estimate of how the system would behave on truly unseen defects.
- **A stronger backbone improves detection but not localization tightness.** Swapping PatchCore's frozen feature extractor from ResNet18 to WideResNet50-2 raised image ROC-AUC from 0.909 to 0.935, but mean IoU/Dice stayed roughly the same — both backbones produce patch features at the same coarse 28×28 grid, so the bilinear-upsampling blur (not backbone capacity) is the bottleneck for tight segmentation.
- **An ensemble can't out-perform a classifier that's already at its evaluation ceiling.** Fusing the supervised classifier with PatchCore's score seemed like an obvious way to get the best of both, but on the classifier's own held-out validation subset the classifier alone already hits AUROC 1.0 (for the same reason its plain validation metrics are inflated — see above), so there's no room left for the ensemble to improve on. A fusion technique can only be shown to add value when tested on a sample that's genuinely novel to *every* component model, not just to one of them.
- **High pixel AUROC does not imply tight defect segmentation.** Mean IoU/Dice came out low (~0.04/0.07) even though pixel AUROC was high. The anomaly map is produced on a coarse 28×28 feature grid and bilinearly upsampled to 224×224, so it's good at *ranking* defect pixels highly (localizing the general region) but blurry compared to MVTec's tight ground-truth masks. Pixel AUROC and IoU/Dice answer different questions and should be reported together, not interchangeably.
- **`.gitignore` patterns without a leading `/` match at any depth.** An earlier `data/` rule (meant to exclude the raw MVTec dataset) was silently also excluding `src/data/`, so real source code was never staged. Always sanity-check ignore rules with `git check-ignore -v <path>` and `git status --ignored` before trusting `git add -A`.
- **Virtual environments need explicit, exact ignore entries.** `.venv/` in `.gitignore` did not match a second environment folder named `.venv-1/`, which had all project dependencies installed — a `git add -A` would have swept hundreds of MB of installed packages into the repo if left unnoticed.
- **Without foreground masking, the anomaly heatmap bleeds into the background.** Since PatchCore scores every patch (including plain background), and each image's heatmap is min-max normalized independently, small background texture/contrast differences got stretched into visible red/yellow — mimicking a real defect signal even though nothing was wrong there. Adding a simple Otsu-threshold foreground mask ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py)) to suppress background patches (pinning them to the object's minimum anomaly distance) fixed this without hurting detection accuracy (image AUROC 0.91 → 0.909, pixel AUROC 0.973 → 0.976) — a good reminder that per-image score normalization can amplify noise anywhere the model doesn't explicitly ignore it.
- **A plain min-max colored heatmap doesn't visually match the actual decision boundary.** Even with background masked out, ordinary (sub-threshold) screw texture — like normal thread ridges — still has a non-zero, spatially-varying anomaly score, so a full-range jet colormap can render it yellow/green and look like "a big defect" even when the real decision (score vs. the calibrated threshold) says otherwise. Anchoring the colormap to the decision threshold (`normalize_map_threshold` in [src/visualization/heatmap.py](src/visualization/heatmap.py)) — compressing below-threshold values into the cool half and only letting above-threshold values read as hot — makes the heatmap visually agree with the Normal/Defective/severity verdict.
- **`matplotlib.imshow` silently re-normalizes data unless you pass `vmin`/`vmax`.** After pre-compressing anomaly values into a threshold-anchored `[0, 1]` range, `axes.imshow(normalized, cmap="jet")` was auto-rescaling that already-compressed range back to the full colormap, quietly undoing the fix for one figure panel. Any time you pass pre-normalized data to `imshow`, pass `vmin=0, vmax=1` explicitly or matplotlib will stretch contrast based on the data's own min/max.
- **A vision heuristic tuned on one category can silently break another.** The foreground-masking fix that helped `screw` (and generalized fine to `hazelnut`) made `bottle`'s pixel-level localization *worse than random* until it was made a per-category, config-driven toggle instead of an always-on assumption — see [Generalization to Other Categories](#generalization-to-other-categories) above. Multi-category testing caught this; single-category testing would not have.
- **Recognizing the object type is a much easier task than recognizing its defects.** The category classifier ([src/models/train_category_classifier.py](src/models/train_category_classifier.py)) hits 100% validation accuracy telling screw/bottle/hazelnut/carpet/leather apart, in contrast to the earlier "misleadingly perfect" good-vs-defective classifier finding. This is expected and not a red flag the same way: whole object types differ enormously in shape/texture/color (an easy, well-separated classification problem), while a defect is a subtle local deviation within one object type (a hard, fine-grained problem) — a perfect score means something very different depending on which of the two problems is being solved.
- **A lesson learned from one category, once turned into a config option, actually transfers.** Adding `carpet` and then `leather` (both full-frame textures like `bottle`) with `use_foreground_mask: false` set from the start — instead of rediscovering the problem — produced the best pixel-level ROC-AUC so far each time (carpet 0.987, then leather 0.991). Turning a bug fix into an explicit, per-category config decision (rather than just patching the one case that broke) is what makes a lesson actually reusable on the next category.
- **A from-scratch CNN needs far more training than a pretrained backbone on a small dataset.** Adding a `simple_cnn` option (small sequential CNN, no pretrained weights) to [src/models/baseline_classifier.py](src/models/baseline_classifier.py) and training it side-by-side with ResNet18/EfficientNet-B0 on the same 112 images/10 epochs showed all three reach the same (misleadingly perfect) validation score, but their training loss tells a very different story: ResNet18/EfficientNet-B0 converge to ~0.01–0.04 while the from-scratch CNN is still at ~0.17. When validation metrics saturate/tie across models (often a sign the eval set is too small or too easy), check the training loss curve too — it can reveal a real gap that accuracy alone hides.
- **`torch.cuda.is_available()` alone misses Apple Silicon GPUs.** The training scripts ([train_baseline.py](src/models/train_baseline.py), [train_category_classifier.py](src/models/train_category_classifier.py), [run_anomaly_detection.py](src/models/run_anomaly_detection.py)) only checked for CUDA and silently fell back to CPU on this Mac, even though `torch.backends.mps.is_available()` was `True`. Adding an explicit `cuda` → `mps` → `cpu` fallback let the category classifier retrain (1631 images, 8 epochs) run on the Apple Silicon GPU instead of CPU with no code/behavior change beyond speed. Always check for `mps` explicitly on Apple Silicon — `cuda.is_available()` being `False` doesn't mean no GPU is available.
- **A fine-grained per-category defect-type classifier is a fundamentally harder problem than the coarse good/defective or category classifiers, and doesn't automatically inherit their near-perfect scores.** See [Defect-Type Classification](#defect-type-classification-per-category) above: screw scored only 0.444 validation accuracy across its 5 defect types (~16–17 train images/class, and the defect types are subtle geometric variations of each other) while leather scored 0.964 with a similar amount of data but visually distinct defect types. Class count and dataset size alone don't predict fine-grained accuracy — inter-class visual similarity matters just as much, and should be checked per category rather than assumed to generalize from one category's result.
