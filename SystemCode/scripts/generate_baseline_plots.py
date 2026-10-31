"""Generate comparison plots for baseline classifiers."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

CATEGORIES = ["screw", "hazelnut", "transistor", "wood", "tile"]
ARCHITECTURES = ["resnet18", "efficientnet_b0", "simple_cnn"]

def generate_loss_accuracy_plots():
    """Generate loss and accuracy curves for all categories × architectures."""
    
    for category in CATEGORIES:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        fig.suptitle(f"Baseline Classifier Comparison: {category.upper()}\nLoss and Accuracy Curves", 
                     fontsize=16, fontweight='bold')
        
        for idx, arch in enumerate(ARCHITECTURES):
            metrics_file = METRICS_DIR / f"baseline_{arch}_{category}_metrics.json"
            
            if not metrics_file.exists():
                print(f"⚠️  Missing: {metrics_file}")
                continue
            
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # If no history, skip plotting
            if "history" not in metrics:
                print(f"ℹ️  No history in: {metrics_file}")
                continue
            
            history = metrics["history"]
            epochs = [h["epoch"] for h in history]
            train_loss = [h["train_loss"] for h in history]
            val_loss = [h["val_loss"] for h in history]
            train_acc = [h["train_accuracy"] for h in history]
            val_acc = [h["val_accuracy"] for h in history]
            
            # Loss subplot
            ax_loss = fig.add_subplot(gs[0, idx])
            ax_loss.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=4)
            ax_loss.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2, markersize=4)
            ax_loss.set_title(f"{arch.upper()}\nLoss", fontweight='bold')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend(loc='best')
            ax_loss.grid(True, alpha=0.3)
            
            # Accuracy subplot
            ax_acc = fig.add_subplot(gs[1, idx])
            ax_acc.plot(epochs, train_acc, 'o-', label='Train Accuracy', linewidth=2, markersize=4)
            ax_acc.plot(epochs, val_acc, 's-', label='Val Accuracy', linewidth=2, markersize=4)
            ax_acc.set_title(f"{arch.upper()}\nAccuracy", fontweight='bold')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_ylim([0, 1.05])
            ax_acc.legend(loc='best')
            ax_acc.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = FIGURES_DIR / f"baseline_comparison_loss_accuracy_{category}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

def generate_confusion_matrix_plots():
    """Generate confusion matrix plots for all categories × architectures."""
    
    for category in CATEGORIES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Confusion Matrices: {category.upper()}", fontsize=14, fontweight='bold')
        
        for idx, arch in enumerate(ARCHITECTURES):
            metrics_file = METRICS_DIR / f"baseline_{arch}_{category}_metrics.json"
            
            if not metrics_file.exists():
                print(f"⚠️  Missing: {metrics_file}")
                continue
            
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            cm = np.array(metrics["confusion_matrix"])
            
            # Normalize for better visualization
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            ax = axes[idx]
            im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                           ha='center', va='center', color='white' if cm_norm[i, j] > 0.5 else 'black',
                           fontsize=10, fontweight='bold')
            
            ax.set_title(f"{arch.upper()}", fontweight='bold')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Defective'])
            ax.set_yticklabels(['Normal', 'Defective'])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Normalized Count')
        
        plt.tight_layout()
        output_path = FIGURES_DIR / f"baseline_comparison_confusion_matrices_{category}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

def main():
    print("Generating loss/accuracy plots...")
    generate_loss_accuracy_plots()
    print("\nGenerating confusion matrix plots...")
    generate_confusion_matrix_plots()
    print("\n✅ All plots generated!")

if __name__ == "__main__":
    main()
