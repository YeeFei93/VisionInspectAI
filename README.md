# VisionInspectAI
Use the MVTec-AD category to detect whether an image is normal or defective, and show the defect location using a heatmap.

## Learning & Techniques Involved

This project combines multiple pattern recognition / machine learning aspects on the same MVTec-AD `screw` category dataset:

- **Supervised / unsupervised learning scenarios**
  - Supervised: [src/models/baseline_classifier.py](src/models/baseline_classifier.py) — a ResNet18/EfficientNet-B0 image classifier fine-tuned on labeled good/defective images (trained via [src/models/train_baseline.py](src/models/train_baseline.py)).
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
  - The [app/streamlit_app.py](app/streamlit_app.py) demo turns an uploaded image into an actionable decision: Normal/Defective prediction, anomaly score, severity (Low/Medium/High), and a heatmap overlay of the likely defect region.

## Project Structure

- `src/data/` — manifest generation ([create_manifest.py](src/data/create_manifest.py)) and PyTorch `Dataset` ([dataset.py](src/data/dataset.py))
- `src/preprocessing/` — image transforms
- `src/models/` — baseline classifier, PatchCore anomaly detector, and training/evaluation scripts
- `src/evaluation/` — classification and pixel-level localization metrics
- `src/visualization/` — anomaly heatmap rendering
- `notebooks/` — dataset exploration
- `app/` — Streamlit inspection demo
- `config/` — per-category YAML configs (e.g. `screw_config.yaml`)

## Quick Start

```bash
# 1. Build the manifest CSV for a category
python -m src.data.create_manifest --category screw

# 2. Train the supervised baseline classifier
python -m src.models.train_baseline --config config/screw_config.yaml

# 3. Train and evaluate the unsupervised PatchCore anomaly detector
python -m src.models.run_anomaly_detection --config config/screw_config.yaml

# 4. Launch the interactive demo
streamlit run app/streamlit_app.py
```

## Notes & Lessons Learned

- **The baseline supervised classifier's near-perfect scores are misleading.** Since MVTec's `train/` only contains `good` images, a supervised good-vs-defective classifier has to be trained on a split carved out of `test/` — meaning every defect *type* it's evaluated on was already seen during training. This produced accuracy/precision/recall/F1 all at 1.0, which reflects a tiny, easy, non-independent validation split rather than real-world generalization. Treat this model as a baseline/sanity-check only.
- **The unsupervised PatchCore detector is the more trustworthy signal.** Trained only on `train/good` (no labels at all) and evaluated on the *entire* labeled test set, it scored a more believable 0.91 image-level ROC-AUC and 0.97 pixel-level AUROC — a much fairer estimate of how the system would behave on truly unseen defects.
- **High pixel AUROC does not imply tight defect segmentation.** Mean IoU/Dice came out low (~0.04/0.07) even though pixel AUROC was high. The anomaly map is produced on a coarse 28×28 feature grid and bilinearly upsampled to 224×224, so it's good at *ranking* defect pixels highly (localizing the general region) but blurry compared to MVTec's tight ground-truth masks. Pixel AUROC and IoU/Dice answer different questions and should be reported together, not interchangeably.
- **`.gitignore` patterns without a leading `/` match at any depth.** An earlier `data/` rule (meant to exclude the raw MVTec dataset) was silently also excluding `src/data/`, so real source code was never staged. Always sanity-check ignore rules with `git check-ignore -v <path>` and `git status --ignored` before trusting `git add -A`.
- **Virtual environments need explicit, exact ignore entries.** `.venv/` in `.gitignore` did not match a second environment folder named `.venv-1/`, which had all project dependencies installed — a `git add -A` would have swept hundreds of MB of installed packages into the repo if left unnoticed.
- **Without foreground masking, the anomaly heatmap bleeds into the background.** Since PatchCore scores every patch (including plain background), and each image's heatmap is min-max normalized independently, small background texture/contrast differences got stretched into visible red/yellow — mimicking a real defect signal even though nothing was wrong there. Adding a simple Otsu-threshold foreground mask ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py)) to suppress background patches (pinning them to the object's minimum anomaly distance) fixed this without hurting detection accuracy (image AUROC 0.91 → 0.909, pixel AUROC 0.973 → 0.976) — a good reminder that per-image score normalization can amplify noise anywhere the model doesn't explicitly ignore it.
- **A plain min-max colored heatmap doesn't visually match the actual decision boundary.** Even with background masked out, ordinary (sub-threshold) screw texture — like normal thread ridges — still has a non-zero, spatially-varying anomaly score, so a full-range jet colormap can render it yellow/green and look like "a big defect" even when the real decision (score vs. the calibrated threshold) says otherwise. Anchoring the colormap to the decision threshold (`normalize_map_threshold` in [src/visualization/heatmap.py](src/visualization/heatmap.py)) — compressing below-threshold values into the cool half and only letting above-threshold values read as hot — makes the heatmap visually agree with the Normal/Defective/severity verdict.
- **`matplotlib.imshow` silently re-normalizes data unless you pass `vmin`/`vmax`.** After pre-compressing anomaly values into a threshold-anchored `[0, 1]` range, `axes.imshow(normalized, cmap="jet")` was auto-rescaling that already-compressed range back to the full colormap, quietly undoing the fix for one figure panel. Any time you pass pre-normalized data to `imshow`, pass `vmin=0, vmax=1` explicitly or matplotlib will stretch contrast based on the data's own min/max.
