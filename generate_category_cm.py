import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from pathlib import Path

# Load metrics
metrics_path = Path("outputs/metrics/category_classifier_metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)

# Get the confusion matrix and categories
cm = np.array(metrics.get("confusion_matrix", metrics.get("test_confusion_matrix", [])))
categories = metrics["categories"]

print(f"Confusion matrix shape: {cm.shape}")
print(f"Categories: {categories}")

# Create the figure
side = max(4.0, 0.5 * len(categories) + 2.0)
figsize = (side, side)
fig, ax = plt.subplots(figsize=figsize)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
    xticks_rotation="vertical" if len(categories) > 4 else "horizontal",
)
ax.set_title("Category classifier — confusion matrix (9 categories)")
fig.tight_layout()

output_path = Path("outputs/figures/category_classifier_resnet18_confusion_matrix.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
print(f"Saved confusion matrix to {output_path}")
plt.close()
