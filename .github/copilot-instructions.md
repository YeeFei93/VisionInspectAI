# VisionInspectAI — Copilot Instructions

MVTec-AD anomaly detection/classification pipeline (screw, bottle, hazelnut, carpet categories) with a Streamlit demo. See [README.md](../README.md) for the full pipeline write-up, model comparisons, and lessons learned — read it before making non-trivial changes.

## Course Context (NUS-ISS Practice Module)

This project is the practice module project for the NUS-ISS Graduate Certificate in Pattern Recognition Systems. Assessment requires the project to demonstrate **at least 3 of these 4 aspects** (already satisfied — see [README.md § Learning & Techniques Involved](../README.md#learning--techniques-involved)): supervised/unsupervised learning scenarios, ML/deep learning techniques, hybrid ML/ensemble approach, intelligent sensing/sense making. Required deliverables: a runnable system, datasets, a final report (tools/techniques used, system design/models, system performance, findings and discussions), code/model files, a 10–15 min video presentation, two presentation slide decks, and a 1–2 page individual reflection report per member. Don't remove or weaken evidence of any of the 4 aspects above without discussing it with the user first.

## Architecture

- `src/data/create_manifest.py` scans `data/mvtec_anomaly_detection/<category>/` and writes `data/manifests/<category>.csv` (the single source of truth every downstream script reads).
- `src/models/train_baseline.py` trains a supervised good/defective classifier (`resnet18` / `efficientnet_b0` / `simple_cnn`, see `src/models/baseline_classifier.py`) on a train/val split carved out of the labeled `test/` rows (MVTec's `train/` only has `good` images).
- `src/models/run_anomaly_detection.py` fits an unsupervised PatchCore detector (`src/models/anomaly_detector.py`) on `train/good` only, then evaluates on the full labeled test set.
- `src/models/train_category_classifier.py` trains a ResNet18 to auto-detect which category an image belongs to; `app/streamlit_app.py` uses it to route to the right per-category PatchCore model.
- `src/models/train_defect_classifier.py` trains a per-category classifier over `defect_type` labels (defective images only, e.g. leather: `color`/`cut`/`fold`/`glue`/`poke`) so the Streamlit demo can show what kind of defect was found, not just Normal/Defective. Optional per category — the app gracefully skips this if a category has no trained checkpoint.
- `src/models/run_ensemble.py` fuses the classifier + PatchCore scores.
- Everything is driven by per-category YAML in `config/` (e.g. `screw_config.yaml`) — copy one to add a new category, don't hardcode category-specific values in code.

## Build, Test, Run

```bash
pip install -r requirement.txt        # note: filename is "requirement.txt", not "requirements.txt"
python -m src.data.create_manifest --category <cat>
python -m src.models.train_baseline --config config/<cat>_config.yaml
python -m src.models.run_anomaly_detection --config config/<cat>_config.yaml
python -m src.models.train_category_classifier --categories screw bottle hazelnut carpet
streamlit run app/streamlit_app.py
python -m pytest tests/ -v
```

## Conventions

- Output artifacts are named `<method>_<architecture>_<category>` (e.g. `baseline_resnet18_screw`, `patchcore_wide_resnet50_2_screw`) so different architectures/categories never overwrite each other's checkpoint (`models/checkpoints/`) or metrics (`outputs/metrics/`) files.
- `data/mvtec_anomaly_detection/`, `models/checkpoints/`, and `outputs/` are all gitignored (large/generated) — only source code and `data/manifests/*.csv` are tracked. Don't assume these exist; scripts must be (re)run locally.
- `.gitignore` uses `/data/*` (not `data/`) specifically because a leading-slash-less rule previously and silently excluded `src/data/` too — keep ignore rules anchored (`/path`) unless a pattern is truly meant to match at any depth.
- `use_foreground_mask` (per-category YAML flag, plumbed through `run_anomaly_detection.py` and the Streamlit app) must stay per-category config, not a global default — it helps categories with a plain background (screw, hazelnut) but actively breaks full-frame textures (bottle, carpet). When adding a category, decide this explicitly rather than assuming the previous category's setting.
- Tests (`tests/`) only cover pure, model-free functions (metrics, heatmap normalization, segmentation) so they run without any trained checkpoints — keep new tests dependency-free in the same way.

## Documenting Findings (Academic Record)

This is a learning/academic project — [README.md](../README.md) doubles as the lab notebook, not just usage docs. Whenever a change produces a new result (a training run, a new category, a bug fix that changes metrics, a comparison between configs/architectures/backbones), update README.md in the same change:

- Add/update a row in the relevant results table (`Model Comparison`, `PatchCore backbone` comparison, or `Generalization to Other Categories`) with the concrete numbers (accuracy/F1, ROC-AUC, IoU/Dice, etc.) — don't just describe results in prose without the metrics.
- Add a bullet to **Notes & Lessons Learned** when something surprising, counter-intuitive, or previously-wrong-assumption is discovered (e.g. a heuristic that didn't generalize, a metric that was misleading, a root cause behind an unexpected number). State the finding, the evidence (numbers), and the takeaway — follow the existing bullets' style as a template.
- Prefer editing existing tables/bullets over duplicating them; only add a new subsection if the finding doesn't fit an existing one.
- Keep it factual and evidence-based: cite the actual metric values produced by the run, not estimates.
