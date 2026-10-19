"""Repeated stratified K-fold CV for the defect-type routing and ensemble decision.

Every outer fold trains each architecture on the training fold only. Early stopping
uses an inner stratified split of that training fold, so the held-out fold is never
used for checkpoint selection (unlike the single-split benchmarks). Out-of-fold
probabilities are saved per fold so strategies can be compared and routing can be
re-selected without refitting.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset
from src.models.baseline_classifier import build_baseline_model
from src.models.benchmark_defect_classifiers import (
    VariantDataset,
    expand_training_rows,
    format_elapsed,
    predict_probabilities,
)
from src.models.run_defect_classifier_ensembles import seed_everything
from src.models.train_defect_classifier import PROJECT_ROOT, build_defect_manifest, train_with_early_stopping
from src.preprocessing.transform import get_val_transforms

ALL_ARCHITECTURES = ("convnext_tiny", "resnet18", "efficientnet_b0", "densenet121")
ROUTING_ARCHITECTURES = ("convnext_tiny", "densenet121")
INNER_VAL_FRACTION = 0.15
DEFAULT_OUTPUT_DIR = Path("outputs/metrics/defect_classifier_cv")


def fold_path(output_dir, category, repeat, fold):
    return output_dir / f"{category}_repeat{repeat}_fold{fold}.json"


def run_fold(category, config, manifest, defect_types, train_idx, test_idx, architectures, seed, device):
    data_cfg, train_cfg = config["data"], config["train"]
    image_size = data_cfg["image_size"]
    train_rows = manifest.iloc[train_idx].reset_index(drop=True)
    test_rows = manifest.iloc[test_idx].reset_index(drop=True)
    fit_rows, inner_rows = train_test_split(
        train_rows, test_size=INNER_VAL_FRACTION, random_state=seed, stratify=train_rows["label"]
    )
    fit_rows = fit_rows.reset_index(drop=True)
    inner_rows = inner_rows.reset_index(drop=True)

    def loader(rows):
        dataset = ManifestImageDataset(rows, PROJECT_ROOT, transform=get_val_transforms(image_size))
        return DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=False,
                          num_workers=train_cfg["num_workers"])

    inner_loader, test_loader = loader(inner_rows), loader(test_rows)
    train_dataset = VariantDataset(expand_training_rows(fit_rows, category), image_size)

    probabilities = {}
    for architecture in architectures:
        seed_everything(seed)
        train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                                  num_workers=train_cfg["num_workers"])
        model = build_baseline_model(
            architecture=architecture,
            num_classes=len(defect_types),
            pretrained=config["model"]["pretrained"],
        ).to(device)
        started = time.perf_counter()
        _, _, summary = train_with_early_stopping(
            model, train_loader, inner_loader, train_cfg, train_cfg["learning_rate"],
            int(train_cfg["epochs"]), device,
        )
        probs, y_true = predict_probabilities(model, test_loader, device)
        probabilities[architecture] = probs.tolist()
        print(f"  {category} {architecture} best_epoch={summary['best_epoch']} "
              f"fold_acc={float((probs.argmax(1) == y_true).mean()):.3f} "
              f"elapsed={format_elapsed(time.perf_counter() - started)}", flush=True)
    return {
        "category": category,
        "defect_types": defect_types,
        "image_paths": test_rows["image_path"].tolist(),
        "y_true": [int(v) for v in y_true],
        "probabilities": probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-ensemble", action="store_true",
                        help="Also train ResNet18 and EfficientNet-B0 so the 4-model votes can be scored")
    parser.add_argument("--torch-threads", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load((PROJECT_ROOT / "config" / f"{args.category}_config.yaml").read_text(encoding="utf-8"))
    manifest, defect_types = build_defect_manifest(PROJECT_ROOT / config["data"]["manifest_path"])
    architectures = ALL_ARCHITECTURES if args.full_ensemble else ROUTING_ARCHITECTURES
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for repeat in range(args.repeats):
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(manifest, manifest["label"])):
            path = fold_path(output_dir, args.category, repeat, fold)
            if path.exists():
                print(f"skip {path.name} (exists)", flush=True)
                continue
            print(f"{args.category} repeat={repeat} fold={fold} test={len(test_idx)} "
                  f"elapsed={format_elapsed(time.perf_counter() - start)}", flush=True)
            result = run_fold(args.category, config, manifest, defect_types, train_idx, test_idx,
                              architectures, args.seed + repeat * 100 + fold, device)
            path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
