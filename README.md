# VisionInspectAI
Use the MVTec-AD category to detect whether an image is normal or defective, and show the defect location using a heatmap.

## Learning & Techniques Involved

This project combines multiple pattern recognition / machine learning aspects on the same MVTec-AD `screw` category dataset:

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
  - The [app/streamlit_app.py](app/streamlit_app.py) demo turns an uploaded image into an actionable decision: Normal/Defective prediction, anomaly score, severity (Low/Medium/High), and a heatmap overlay of the likely defect region.

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

# 4. Launch the interactive demo
streamlit run app/streamlit_app.py

# 5. Run the unit tests
python -m pytest tests/ -v
```

To compare baseline classifier architectures, point `train_baseline.py` at any of `config/screw_config.yaml` (ResNet18), `config/screw_config_efficientnet_b0.yaml`, or `config/screw_config_simple_cnn.yaml` — each writes its own checkpoint/metrics file so results don't overwrite each other.

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
- **A from-scratch CNN needs far more training than a pretrained backbone on a small dataset.** Adding a `simple_cnn` option (small sequential CNN, no pretrained weights) to [src/models/baseline_classifier.py](src/models/baseline_classifier.py) and training it side-by-side with ResNet18/EfficientNet-B0 on the same 112 images/10 epochs showed all three reach the same (misleadingly perfect) validation score, but their training loss tells a very different story: ResNet18/EfficientNet-B0 converge to ~0.01–0.04 while the from-scratch CNN is still at ~0.17. When validation metrics saturate/tie across models (often a sign the eval set is too small or too easy), check the training loss curve too — it can reveal a real gap that accuracy alone hides.
