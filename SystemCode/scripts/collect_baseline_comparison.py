"""Collect baseline classifier metrics and create comparison tables."""
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# Categories and architectures to compare
CATEGORIES = ["screw", "hazelnut", "transistor", "wood", "tile"]
ARCHITECTURES = ["resnet18", "efficientnet_b0", "simple_cnn", "convnext_tiny", "vit_b_16"]

def collect_metrics():
    """Collect metrics from all baseline runs."""
    results = []
    
    for category in CATEGORIES:
        for arch in ARCHITECTURES:
            metrics_file = METRICS_DIR / f"baseline_{arch}_{category}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                final_train_loss = None
                if "history" in metrics and metrics["history"]:
                    final_train_loss = metrics["history"][-1]["train_loss"]
                
                best_epoch = metrics.get("best_epoch", "N/A")
                epochs_trained = metrics.get("epochs_trained", "N/A")
                
                results.append({
                    "Category": category,
                    "Architecture": arch,
                    "Final Train Loss": final_train_loss,
                    "Val Accuracy": metrics["accuracy"],
                    "Val Precision": metrics["precision"],
                    "Val Recall": metrics["recall"],
                    "Val F1": metrics["f1"],
                    "Best Epoch": best_epoch,
                    "Epochs Trained": epochs_trained,
                })
            else:
                print(f"⚠️  Missing: {metrics_file}")
    
    return pd.DataFrame(results)

def main():
    df = collect_metrics()
    
    print("\n" + "="*120)
    print("BASELINE CLASSIFIER COMPARISON: 5 Categories × 3 Architectures")
    print("="*120 + "\n")
    
    # Full table
    print(df.to_string(index=False))
    print("\n")
    
    # Summary by architecture
    print("="*80)
    print("SUMMARY BY ARCHITECTURE (Mean ± Std across 5 categories)")
    print("="*80 + "\n")
    
    arch_summary = df.groupby("Architecture").agg({
        "Final Train Loss": ["mean", "std"],
        "Val Accuracy": ["mean", "std"],
        "Val F1": ["mean", "std"],
    }).round(4)
    
    print(arch_summary)
    print("\n")
    
    # Summary by category
    print("="*80)
    print("SUMMARY BY CATEGORY (Comparison across 3 architectures)")
    print("="*80 + "\n")
    
    for category in CATEGORIES:
        cat_data = df[df["Category"] == category].sort_values("Val Accuracy", ascending=False)
        print(f"\n{category.upper()}:")
        print(cat_data[["Architecture", "Final Train Loss", "Val Accuracy", "Val F1"]].to_string(index=False))
    
    # Ranking
    print("\n\n" + "="*80)
    print("RANKING: Best Val Accuracy by Category")
    print("="*80 + "\n")
    
    for category in CATEGORIES:
        best = df[df["Category"] == category].nlargest(1, "Val Accuracy").iloc[0]
        print(f"{category:12} -> {best['Architecture']:15} ({best['Val Accuracy']:.2%}, F1={best['Val F1']:.4f})")
    
    # Overall ranking
    print("\n\n" + "="*80)
    print("OVERALL RANKING: Average Val Accuracy across all categories")
    print("="*80 + "\n")
    
    overall = df.groupby("Architecture")["Val Accuracy"].mean().sort_values(ascending=False)
    for i, (arch, acc) in enumerate(overall.items(), 1):
        print(f"{i}. {arch:15} -> {acc:.2%}")
    
    # Save to CSV
    csv_path = PROJECT_ROOT / "outputs" / "baseline_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

if __name__ == "__main__":
    main()
