---
description: "Draft/update the NUS-ISS Practice Module final project report from README.md and current metrics"
agent: "agent"
---
Draft or update the final project report deliverable required by the NUS-ISS Practice Module (Graduate Certificate in Pattern Recognition Systems), for this project.

Write it to `docs/final_report.md` (create the `docs/` folder if needed; if the file already exists, update it in place rather than duplicating sections).

Ground everything in the actual codebase and [README.md](../../README.md) — do not invent numbers or claims. Pull concrete metrics from `outputs/metrics/*.json` where available, and from the README's results tables otherwise.

Required structure (per the assessment brief):

1. **Introduction** — the real-world problem being solved (MVTec-AD visual defect inspection), and which of the 4 required aspects (supervised/unsupervised learning, ML/DL techniques, hybrid/ensemble approach, intelligent sensing/sense making) the project demonstrates — reference [README.md § Learning & Techniques Involved](../../README.md#learning--techniques-involved).
2. **Tools/techniques used** — summarize the stack (PyTorch/torchvision, scikit-learn, OpenCV, Streamlit) and the algorithms (ResNet18/EfficientNet-B0/SimpleCNN classifiers, PatchCore anomaly detection with greedy coreset subsampling, Otsu foreground segmentation, category auto-detection).
3. **System design / Models** — architecture summary (manifest → baseline classifier → PatchCore → category classifier → ensemble → Streamlit demo), referencing [README.md § Project Structure](../../README.md#project-structure) and [README.md § Pipeline Steps in Detail](../../README.md#pipeline-steps-in-detail).
4. **System performance** — the actual metrics tables (Model Comparison, PatchCore backbone comparison, Generalization to Other Categories, Hybrid Ensemble) copied/summarized from README.md, kept in sync with it.
5. **Findings and discussions** — condense the README's "Notes & Lessons Learned" bullets into a discussion section, keeping the evidence (numbers) and takeaway for each.

Keep it concise and factual; link back to README.md sections rather than duplicating large blocks of prose.
