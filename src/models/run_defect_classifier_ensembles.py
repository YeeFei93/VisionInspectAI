"""Train and compare voting and ConvNeXt bagging for defect-type classifiers."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import ManifestImageDataset
from src.models.baseline_classifier import build_baseline_model
from src.models.benchmark_defect_classifiers import (
    CATEGORIES,
    VariantDataset,
    classification_scores,
    expand_training_rows,
    format_elapsed,
    predict_probabilities,
)
from src.models.train_defect_classifier import (
    PROJECT_ROOT,
    build_defect_manifest,
    train_with_early_stopping,
)
from src.preprocessing.transform import get_val_transforms

ARCHITECTURES = ("resnet18", "efficientnet_b0", "densenet121")
CONVNEXT_BAGGING_SEEDS = (42, 43, 44)
WEIGHTED_VOTE_WEIGHTS = (0.25, 0.50, 0.75)
OUTPUT_PREFIX = "defect_classifier_ensemble"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_bootstrap_rows(rows, seed):
    sampled_classes = []
    for class_index, (_, class_rows) in enumerate(rows.groupby("label", sort=True)):
        sampled_classes.append(
            class_rows.sample(
                n=len(class_rows),
                replace=True,
                random_state=seed + class_index,
            )
        )
    return pd.concat(sampled_classes).reset_index(drop=True)


def hard_vote_probabilities(probabilities_by_model):
    model_probabilities = list(probabilities_by_model.values())
    individual_predictions = np.stack(
        [probabilities.argmax(axis=1) for probabilities in model_probabilities]
    )
    n_samples, n_classes = model_probabilities[0].shape
    votes = np.zeros((n_samples, n_classes), dtype=np.int64)
    for predictions in individual_predictions:
        np.add.at(votes, (np.arange(n_samples), predictions), 1)

    mean_probabilities = np.mean(model_probabilities, axis=0)
    majority_classes = votes == votes.max(axis=1, keepdims=True)
    tied_probabilities = np.where(majority_classes, mean_probabilities, 0.0)
    return tied_probabilities / tied_probabilities.sum(axis=1, keepdims=True)


def weighted_soft_vote(probabilities_by_model, weights):
    if set(probabilities_by_model) != set(weights):
        raise ValueError("Each model must have exactly one voting weight")
    model_names = list(probabilities_by_model)
    model_weights = np.asarray([weights[name] for name in model_names], dtype=float)
    if np.any(model_weights < 0) or model_weights.sum() <= 0:
        raise ValueError("Voting weights must be non-negative with a positive sum")
    model_weights /= model_weights.sum()
    stacked = np.stack([probabilities_by_model[name] for name in model_names])
    return np.average(stacked, axis=0, weights=model_weights)


def build_ensemble_strategies(probabilities_by_model, convnext_bagged, category):
    model_order = ("convnext_tiny", "resnet18", "efficientnet_b0", "densenet121")
    strategies = {
        "four_model_hard_vote": (
            hard_vote_probabilities(
                {name: probabilities_by_model[name] for name in model_order}
            ),
            {"members": ",".join(model_order), "tie_break": "mean_probability"},
        ),
        "four_model_equal_soft_vote": (
            np.mean([probabilities_by_model[name] for name in model_order], axis=0),
            {"members": ",".join(model_order)},
        ),
        "convnext_heavy_soft_vote_70_10_10_10": (
            weighted_soft_vote(
                probabilities_by_model,
                {
                    "convnext_tiny": 0.70,
                    "resnet18": 0.10,
                    "efficientnet_b0": 0.10,
                    "densenet121": 0.10,
                },
            ),
            {
                "members": ",".join(model_order),
                "weights": "convnext_tiny=0.70;others=0.10_each",
            },
        ),
        "convnext_bagging_3_seeds": (
            convnext_bagged,
            {"members": "convnext_tiny", "seeds": "42,43,44"},
        ),
    }
    routed_model = "densenet121" if category in {"bottle", "wood"} else "convnext_tiny"
    strategies["category_routed_single_model"] = (
        probabilities_by_model[routed_model],
        {"members": routed_model, "routed_architecture": routed_model},
    )

    if category in {"bottle", "wood"}:
        for convnext_weight in WEIGHTED_VOTE_WEIGHTS:
            strategy = (
                f"convnext_{int(convnext_weight * 100)}_densenet_"
                f"{int((1.0 - convnext_weight) * 100)}"
            )
            strategies[strategy] = (
                weighted_soft_vote(
                    {
                        "convnext_tiny": probabilities_by_model["convnext_tiny"],
                        "densenet121": probabilities_by_model["densenet121"],
                    },
                    {
                        "convnext_tiny": convnext_weight,
                        "densenet121": 1.0 - convnext_weight,
                    },
                ),
                {
                    "members": "convnext_tiny,densenet121",
                    "weights": (
                        f"convnext_tiny={convnext_weight:.2f};"
                        f"densenet121={1.0 - convnext_weight:.2f}"
                    ),
                },
            )
    return strategies


def make_record(category, strategy, y_true, probabilities, **metadata):
    return {
        "category": category,
        "strategy": strategy,
        **classification_scores(y_true, probabilities),
        **metadata,
    }


def evaluate_predictions(category, strategy, y_true, probabilities, defect_types, image_paths, metadata):
    record = make_record(category, strategy, y_true, probabilities, **metadata)
    predictions = probabilities.argmax(axis=1)
    per_image_records = []
    for index, prediction in enumerate(predictions):
        per_image_records.append(
            {
                "category": category,
                "image_path": image_paths[index],
                "strategy": strategy,
                "true_label": int(y_true[index]),
                "true_defect_type": defect_types[int(y_true[index])],
                "predicted_label": int(prediction),
                "predicted_defect_type": defect_types[int(prediction)],
                "probabilities": json.dumps(probabilities[index].tolist()),
            }
        )
    return record, per_image_records


def load_convnext_seed42(category, config, defect_types, val_loader, device):
    output_cfg = config["output"]
    checkpoint_path = (
        PROJECT_ROOT
        / output_cfg["checkpoint_dir"]
        / f"defect_classifier_convnext_tiny_{category}.pt"
    )
    metadata_path = (
        PROJECT_ROOT
        / output_cfg["metrics_dir"]
        / f"defect_classifier_convnext_tiny_{category}_metrics.json"
    )
    if not checkpoint_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing ConvNeXt-Tiny seed-42 deployment artifacts for {category}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["defect_types"] != defect_types:
        raise ValueError(f"Defect-label order mismatch for {category}")

    model = build_baseline_model(
        architecture="convnext_tiny",
        num_classes=len(defect_types),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    probabilities, y_true = predict_probabilities(model, val_loader, device)
    return probabilities, y_true, metadata


def train_member(
    category,
    architecture,
    seed,
    train_rows,
    val_loader,
    defect_types,
    config,
    device,
    benchmark_start,
    member_index,
    total_members,
    use_bootstrap=False,
):
    data_cfg = config["data"]
    train_cfg = config["train"]
    output_cfg = config["output"]
    seed_everything(seed)
    image_size = data_cfg["image_size"]
    sampled_rows = (
        stratified_bootstrap_rows(train_rows, seed) if use_bootstrap else train_rows
    )
    expanded_rows = expand_training_rows(sampled_rows, category)
    train_dataset = VariantDataset(expanded_rows, image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    model = build_baseline_model(
        architecture=architecture,
        num_classes=len(defect_types),
        pretrained=config["model"]["pretrained"],
    ).to(device)
    epochs = int(train_cfg["epochs"])
    member_start = time.perf_counter()
    print(
        f"PROGRESS category={category} model={member_index}/{total_members} "
        f"{architecture} seed={seed} sampling="
        f"{'stratified_bootstrap' if use_bootstrap else 'original_train_split'} "
        f"status=START "
        f"total_elapsed={format_elapsed(member_start - benchmark_start)}",
        flush=True,
    )
    _, _, training_summary = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        train_cfg,
        train_cfg["learning_rate"],
        epochs,
        device,
    )
    probabilities, y_true = predict_probabilities(model, val_loader, device)

    member_dir = PROJECT_ROOT / output_cfg["checkpoint_dir"] / "ensemble_members"
    metadata_dir = PROJECT_ROOT / output_cfg["metrics_dir"] / "ensemble_members"
    member_name = f"defect_classifier_{architecture}_{category}_seed{seed}"
    checkpoint_path = member_dir / f"{member_name}.pt"
    metadata_path = metadata_dir / f"{member_name}_metrics.json"
    member_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    history = training_summary["history"]
    best_epoch = training_summary["best_epoch"]
    training_record = {
        "category": category,
        "architecture": architecture,
        "seed": seed,
        "train_images": len(train_rows),
        "train_images_with_variants": len(train_dataset),
        "sampling": "stratified_bootstrap" if use_bootstrap else "original_train_split",
        "val_images": len(val_loader.dataset),
        "defect_types": len(defect_types),
        **classification_scores(y_true, probabilities),
        "final_train_loss": float(history[-1]["train_loss"]),
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "model_elapsed_seconds": time.perf_counter() - member_start,
    }
    deployment_metadata = {
        **training_record,
        "defect_types": defect_types,
        "pretrained": bool(config["model"]["pretrained"]),
        "crop_mode": "full_image",
        "image_size": image_size,
    }
    metadata_path.write_text(json.dumps(deployment_metadata, indent=2), encoding="utf-8")
    print(
        f"PROGRESS category={category} model={member_index}/{total_members} "
        f"{architecture} seed={seed} "
        f"sampling={training_record['sampling']} status=DONE "
        f"accuracy={training_record['val_accuracy']:.4f} "
        f"macro_f1={training_record['val_macro_f1']:.4f} "
        f"model_elapsed={format_elapsed(training_record['model_elapsed_seconds'])} "
        f"total_elapsed={format_elapsed(time.perf_counter() - benchmark_start)}",
        flush=True,
    )
    return probabilities, y_true, training_record


def run_category(category, device, benchmark_start):
    config_path = PROJECT_ROOT / "config" / f"{category}_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    train_cfg = config["train"]
    manifest, defect_types = build_defect_manifest(
        PROJECT_ROOT / data_cfg["manifest_path"]
    )
    train_rows, val_rows = train_test_split(
        manifest,
        test_size=data_cfg["val_split"],
        random_state=data_cfg["seed"],
        stratify=manifest["label"],
    )
    train_rows = train_rows.reset_index(drop=True)
    val_rows = val_rows.reset_index(drop=True)
    image_size = data_cfg["image_size"]
    val_dataset = ManifestImageDataset(
        val_rows, PROJECT_ROOT, transform=get_val_transforms(image_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    image_paths = val_rows["image_path"].tolist()
    total_members = 1 + len(ARCHITECTURES) + len(CONVNEXT_BAGGING_SEEDS)
    print(
        f"\n{category}: {len(train_rows)} train / {len(val_rows)} validation; "
        "ConvNeXt seed 42 loaded if available (otherwise trained); "
        f"evaluating {total_members} members",
        flush=True,
    )

    predictions = {}
    run_records = []
    per_image_records = []
    member_index = 1
    output_cfg = config["output"]
    convnext_checkpoint = (
        PROJECT_ROOT
        / output_cfg["checkpoint_dir"]
        / f"defect_classifier_convnext_tiny_{category}.pt"
    )
    convnext_metadata_path = (
        PROJECT_ROOT
        / output_cfg["metrics_dir"]
        / f"defect_classifier_convnext_tiny_{category}_metrics.json"
    )
    if convnext_checkpoint.exists() and convnext_metadata_path.exists():
        member_start = time.perf_counter()
        print(
            f"PROGRESS category={category} model={member_index}/{total_members} "
            f"convnext_tiny seed=42 status=LOAD "
            f"total_elapsed={format_elapsed(member_start - benchmark_start)}",
            flush=True,
        )
        convnext_probs, y_true, convnext_metadata = load_convnext_seed42(
            category, config, defect_types, val_loader, device
        )
        convnext_training_record = None
    else:
        print(
            f"PROGRESS category={category} model={member_index}/{total_members} "
            "convnext_tiny seed=42 status=TRAIN_MISSING_DEPLOYMENT_MODEL",
            flush=True,
        )
        convnext_probs, y_true, convnext_training_record = train_member(
            category,
            "convnext_tiny",
            int(data_cfg["seed"]),
            train_rows,
            val_loader,
            defect_types,
            config,
            device,
            benchmark_start,
            member_index,
            total_members,
        )
        convnext_metadata = None
    predictions["convnext_tiny"] = convnext_probs
    convnext_record_metadata = {"seed": 42}
    if convnext_training_record is not None:
        convnext_record_metadata["final_train_loss"] = convnext_training_record[
            "final_train_loss"
        ]
    record, image_records = evaluate_predictions(
        category,
        "single_convnext_tiny",
        y_true,
        convnext_probs,
        defect_types,
        image_paths,
        convnext_record_metadata,
    )
    run_records.append(record)
    per_image_records.extend(image_records)
    if convnext_training_record is None:
        print(
            f"PROGRESS category={category} model={member_index}/{total_members} "
            f"convnext_tiny seed=42 status=DONE "
            f"model_elapsed={format_elapsed(time.perf_counter() - member_start)} "
            f"total_elapsed={format_elapsed(time.perf_counter() - benchmark_start)}",
            flush=True,
        )
    member_index += 1

    for architecture in ARCHITECTURES:
        probabilities, member_y_true, training_record = train_member(
            category,
            architecture,
            int(data_cfg["seed"]),
            train_rows,
            val_loader,
            defect_types,
            config,
            device,
            benchmark_start,
            member_index,
            total_members,
        )
        if not np.array_equal(y_true, member_y_true):
            raise RuntimeError(f"Validation labels changed for {category}/{architecture}")
        predictions[architecture] = probabilities
        single_record = make_record(
            category,
            f"single_{architecture}",
            y_true,
            probabilities,
            seed=int(data_cfg["seed"]),
            final_train_loss=training_record["final_train_loss"],
        )
        run_records.append(single_record)
        record, image_records = evaluate_predictions(
            category,
            f"single_{architecture}",
            y_true,
            probabilities,
            defect_types,
            image_paths,
            {"seed": int(data_cfg["seed"])},
        )
        per_image_records.extend(image_records)
        member_index += 1

    convnext_members = []
    for seed in CONVNEXT_BAGGING_SEEDS:
        probabilities, member_y_true, training_record = train_member(
            category,
            "convnext_tiny",
            int(seed),
            train_rows,
            val_loader,
            defect_types,
            config,
            device,
            benchmark_start,
            member_index,
            total_members,
            use_bootstrap=True,
        )
        if not np.array_equal(y_true, member_y_true):
            raise RuntimeError(f"Validation labels changed for {category}/convnext_tiny seed={seed}")
        convnext_members.append(probabilities)
        run_records.append(
            {
                "category": category,
                "strategy": "convnext_bagging_member",
                "architecture": "convnext_tiny",
                "seed": seed,
                "sampling": training_record["sampling"],
                **classification_scores(y_true, probabilities),
                "final_train_loss": training_record["final_train_loss"],
            }
        )
        member_index += 1

    strategy_predictions = build_ensemble_strategies(
        predictions, np.mean(convnext_members, axis=0), category
    )
    for strategy, (probabilities, strategy_metadata) in strategy_predictions.items():
        record, image_records = evaluate_predictions(
            category,
            strategy,
            y_true,
            probabilities,
            defect_types,
            image_paths,
            strategy_metadata,
        )
        run_records.append(record)
        per_image_records.extend(image_records)

    return run_records, per_image_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    benchmark_start = time.perf_counter()
    print(f"Device={device}, torch_threads={args.torch_threads}", flush=True)
    print(
        "Ensembles: hard vote, equal soft vote, ConvNeXt-heavy soft vote, "
        "ConvNeXt 3-seed bagging, category-routed single model, and bottle/wood "
        "ConvNeXt-DenseNet weight sweep. Stacking/boosting excluded.",
        flush=True,
    )

    all_runs = []
    all_predictions = []
    for category in args.categories:
        run_records, image_records = run_category(category, device, benchmark_start)
        all_runs.extend(run_records)
        all_predictions.extend(image_records)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(all_runs)
    predictions = pd.DataFrame(all_predictions)
    summary = (
        runs.groupby("strategy", as_index=False)
        .agg(
            categories=("category", "nunique"),
            mean_val_accuracy=("val_accuracy", "mean"),
            std_val_accuracy=("val_accuracy", "std"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            std_val_macro_f1=("val_macro_f1", "std"),
        )
    )
    summary[["std_val_accuracy", "std_val_macro_f1"]] = summary[
        ["std_val_accuracy", "std_val_macro_f1"]
    ].fillna(0.0)
    runs.to_csv(output_dir / f"{args.output_prefix}_runs.csv", index=False)
    summary.to_csv(output_dir / f"{args.output_prefix}_summary.csv", index=False)
    predictions.to_csv(output_dir / f"{args.output_prefix}_per_image_predictions.csv", index=False)
    results = {
        "device": str(device),
        "split": "same seeded stratified 70/30 category split as the ConvNeXt deployment benchmark",
        "validation_caveat": "All metrics are exploratory: validation rows come from MVTec's labeled test split, and voting candidates are compared on the same small validation sets.",
        "seeds": {"single_model_members": 42, "convnext_bagging": [42, 43, 44]},
        "excluded": ["stacking", "boosting"],
        "runs": all_runs,
        "summary": summary.to_dict(orient="records"),
        "total_elapsed": format_elapsed(time.perf_counter() - benchmark_start),
    }
    (output_dir / f"{args.output_prefix}_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nEnsemble strategy summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(
        f"Saved {len(all_runs)} strategy/category rows and {len(all_predictions)} "
        f"per-image predictions; total_elapsed="
        f"{format_elapsed(time.perf_counter() - benchmark_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
