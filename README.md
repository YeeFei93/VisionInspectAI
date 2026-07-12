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
