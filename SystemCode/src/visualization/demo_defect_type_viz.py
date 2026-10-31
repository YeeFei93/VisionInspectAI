"""Demo: Visualize multi-defect-type segmentation with color-coded overlay.

Shows how to use the new defect-type color-coded visualization when an image
has multiple predicted defect types.

Usage:
    python -m src.visualization.demo_defect_type_viz \
        --config config/leather_config.yaml \
        --sample-image data/mvtec_anomaly_detection/leather/test/combined/000.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.dataset import load_manifest
from src.models.anomaly_detector import PatchCoreAnomalyDetector, scoring_artifact_suffix
from src.models.baseline_classifier import build_baseline_model
from src.models.defect_regions import classify_defect_regions, summarize_distinct_types
from src.preprocessing.segmentation import compute_foreground_mask
from src.preprocessing.transform import get_val_transforms
from src.visualization.heatmap import save_defect_type_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/leather_config.yaml"))
    parser.add_argument(
        "--sample-image",
        type=Path,
        help="Path to an image to visualize (relative to project root or absolute)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/defect_type_viz"),
        help="Directory to save visualization",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    category = config.get("category", "screw")
    data_cfg = config["data"]
    anomaly_cfg = config["anomaly_detection"]
    output_cfg = config["output"]
    image_size = data_cfg["image_size"]

    if args.sample_image is None:
        # Find a "combined" defect image from test set as demo
        manifest = load_manifest(PROJECT_ROOT / data_cfg["manifest_path"])
        defective_combined = manifest[
            (manifest["split"] == "test") & (manifest["defect_type"] == "combined")
        ]
        if len(defective_combined) == 0:
            print(f"No 'combined' defect images found in {category}. Using first defective image...")
            defective = manifest[(manifest["split"] == "test") & (manifest["label"] == 1)]
            if len(defective) == 0:
                print(f"No defective images in {category}. Exiting.")
                return
            image_path = PROJECT_ROOT / defective.iloc[0]["image_path"]
        else:
            image_path = PROJECT_ROOT / defective_combined.iloc[0]["image_path"]
    else:
        image_path = args.sample_image if args.sample_image.is_absolute() else PROJECT_ROOT / args.sample_image

    print(f"Loading image: {image_path}")
    original_image = Image.open(image_path).convert("RGB")
    resized_image = original_image.resize((image_size, image_size))
    transform = get_val_transforms(image_size)

    # Load PatchCore detector
    device = torch.device("cpu")
    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=tuple(anomaly_cfg["layers"]),
        device="cpu",
        num_neighbors=anomaly_cfg.get("num_neighbors", 1),
        softmax_reweighting=anomaly_cfg.get("softmax_reweighting", False),
        reweight_num_neighbors=anomaly_cfg.get("reweight_num_neighbors", 9),
    )
    detector_ckpt = (
        PROJECT_ROOT / output_cfg["checkpoint_dir"] / f"patchcore_{anomaly_cfg['backbone']}_{category}_memory_bank.pt"
    )
    if not detector_ckpt.exists():
        print(f"Detector checkpoint not found: {detector_ckpt}")
        print("Run: python -m src.models.run_anomaly_detection --config " + str(args.config))
        return

    detector.load(detector_ckpt)

    # Get PatchCore anomaly map
    print("Running PatchCore anomaly detection...")
    input_tensor = transform(original_image).unsqueeze(0)
    use_foreground_mask = anomaly_cfg.get("use_foreground_mask", True)
    if use_foreground_mask:
        foreground_mask = torch.from_numpy(compute_foreground_mask(resized_image, image_size)).unsqueeze(0)
    else:
        foreground_mask = None
    result = detector.predict(input_tensor, foreground_masks=foreground_mask)[0]
    anomaly_map = result.anomaly_map

    # Load metrics for threshold
    scoring_suffix = scoring_artifact_suffix(
        anomaly_cfg.get("num_neighbors", 1),
        anomaly_cfg.get("softmax_reweighting", False),
        anomaly_cfg.get("reweight_num_neighbors", 9),
    )
    metrics_path = (
        PROJECT_ROOT / output_cfg["metrics_dir"] / f"patchcore_{anomaly_cfg['backbone']}_{category}{scoring_suffix}_metrics.json"
    )
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        pixel_threshold = metrics.get("pixel_threshold", 0.5)
    else:
        pixel_threshold = 0.5

    # Load defect classifier (if available)
    defect_ckpt = (
        PROJECT_ROOT / output_cfg["checkpoint_dir"] / f"defect_classifier_resnet18_{category}.pt"
    )
    if not defect_ckpt.exists():
        print(f"Defect classifier checkpoint not found: {defect_ckpt}")
        print("Run: python -m src.models.train_defect_classifier --config " + str(args.config))
        return

    defect_model = build_baseline_model(
        architecture="resnet18",
        num_classes=None,  # Will be determined from checkpoint
        pretrained=False,
    )

    # Load checkpoint and infer num_classes
    state_dict = torch.load(defect_ckpt, map_location=device)
    # Count classification head output size
    fc_weight_shape = None
    for name, param in state_dict.items():
        if "fc" in name or "classifier" in name:
            if "weight" in name and param.dim() == 2:
                fc_weight_shape = param.shape
                break
    
    if fc_weight_shape:
        num_classes = fc_weight_shape[0]
        defect_model = build_baseline_model(
            architecture="resnet18",
            num_classes=num_classes,
            pretrained=False,
        )
        defect_model.load_state_dict(state_dict)
        defect_model.eval()
        
        # Load defect types from metrics
        defect_metrics_path = (
            PROJECT_ROOT / output_cfg["metrics_dir"] / f"defect_classifier_resnet18_{category}_metrics.json"
        )
        if defect_metrics_path.exists():
            defect_metrics = json.loads(defect_metrics_path.read_text())
            defect_types = defect_metrics.get("defect_types", [])
        else:
            defect_types = [f"type_{i}" for i in range(num_classes)]

        # Classify defect regions
        print("Classifying defect regions...")
        regions = classify_defect_regions(
            resized_image,
            anomaly_map,
            threshold=pixel_threshold,
            defect_model=defect_model,
            defect_types=defect_types,
            padding=8,
            min_area=30,
        )

        if regions:
            # Deduplicate by highest confidence per type
            distinct_regions = summarize_distinct_types(regions)
            print(f"Found {len(distinct_regions)} distinct defect type(s):")
            for region in distinct_regions:
                print(f"  - {region.defect_type}: confidence {region.confidence:.3f}")

            # Save visualization
            output_dir = args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{category}_defect_type_viz.png"
            
            print(f"Saving visualization to {output_path}...")
            save_defect_type_heatmap(
                original_image,
                distinct_regions,
                image_size,
                output_path,
                alpha=0.5,
            )
            print("Done!")
        else:
            print("No anomalous regions found.")
    else:
        print("Could not determine number of classes from checkpoint.")


if __name__ == "__main__":
    main()
