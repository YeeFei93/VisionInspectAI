"""Summarise out-of-fold results from run_defect_classifier_cv.py."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.models.run_defect_classifier_ensembles import hard_vote_probabilities, weighted_soft_vote
from src.models.train_defect_classifier import PROJECT_ROOT

HEAVY_WEIGHTS = {"convnext_tiny": 0.7, "resnet18": 0.1, "efficientnet_b0": 0.1, "densenet121": 0.1}
FULL_MODELS = tuple(HEAVY_WEIGHTS)


def strategy_probabilities(probabilities_by_model):
    """Return {strategy: probability array} for the models available in a fold."""
    probs = {name: np.asarray(p) for name, p in probabilities_by_model.items()}
    strategies = {f"single_{name}": p for name, p in probs.items()}
    if all(name in probs for name in FULL_MODELS):
        ordered = {name: probs[name] for name in FULL_MODELS}
        strategies["four_model_hard_vote"] = hard_vote_probabilities(ordered)
        strategies["four_model_equal_soft_vote"] = weighted_soft_vote(ordered, {n: 1.0 for n in FULL_MODELS})
        strategies["convnext_heavy_soft_vote"] = weighted_soft_vote(ordered, HEAVY_WEIGHTS)
    return strategies


def load_folds(output_dir, category):
    folds = []
    for path in sorted(Path(output_dir).glob(f"{category}_repeat*_fold*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        repeat = int(path.stem.split("_repeat")[1].split("_")[0])
        folds.append((repeat, np.asarray(data["y_true"]), strategy_probabilities(data["probabilities"])))
    return folds


def accuracy(y_true, probabilities):
    return float((probabilities.argmax(1) == y_true).mean())


def macro_f1(y_true, probabilities):
    return float(f1_score(y_true, probabilities.argmax(1), average="macro", zero_division=0))


def paired_counts(folds, strategy_a, strategy_b):
    only_a = only_b = 0
    for _, y, s in folds:
        a_ok = s[strategy_a].argmax(1) == y
        b_ok = s[strategy_b].argmax(1) == y
        only_a += int((a_ok & ~b_ok).sum())
        only_b += int((~a_ok & b_ok).sum())
    return only_a, only_b


def summarise_category(folds):
    names = list(folds[0][2])
    rows = {}
    for name in names:
        per_fold_acc = [accuracy(y, s[name]) for _, y, s in folds]
        y_all = np.concatenate([y for _, y, _ in folds])
        p_all = np.concatenate([s[name] for _, _, s in folds])
        rows[name] = {
            "correct": int((p_all.argmax(1) == y_all).sum()),
            "total": int(len(y_all)),
            "pooled_accuracy": accuracy(y_all, p_all),
            "pooled_macro_f1": macro_f1(y_all, p_all),
            "fold_accuracy_mean": float(np.mean(per_fold_acc)),
            "fold_accuracy_sd": float(np.std(per_fold_acc)),
        }
        by_repeat = {}
        for repeat in sorted({r for r, _, _ in folds}):
            selected = [(y, s[name]) for r, y, s in folds if r == repeat]
            y_repeat = np.concatenate([y for y, _ in selected])
            p_repeat = np.concatenate([p for _, p in selected])
            by_repeat[str(repeat)] = {
                "correct": int((p_repeat.argmax(1) == y_repeat).sum()),
                "total": int(len(y_repeat)),
                "accuracy": accuracy(y_repeat, p_repeat),
                "macro_f1": macro_f1(y_repeat, p_repeat),
            }
        rows[name]["per_repeat"] = by_repeat
        rows[name]["repeat_accuracy_mean"] = float(np.mean([v["accuracy"] for v in by_repeat.values()]))
        rows[name]["repeat_accuracy_sd"] = float(np.std([v["accuracy"] for v in by_repeat.values()]))
        rows[name]["repeat_macro_f1_mean"] = float(np.mean([v["macro_f1"] for v in by_repeat.values()]))
        rows[name]["repeat_macro_f1_sd"] = float(np.std([v["macro_f1"] for v in by_repeat.values()]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/metrics/defect_classifier_cv"))
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "pill"])
    args = parser.parse_args()
    input_dir = PROJECT_ROOT / args.input_dir

    report = {}
    for category in args.categories:
        folds = load_folds(input_dir, category)
        if not folds:
            continue
        summary = summarise_category(folds)
        comparisons = {}
        for repeat in sorted({r for r, _, _ in folds}):
            repeat_folds = [fold for fold in folds if fold[0] == repeat]
            comparisons[str(repeat)] = {
                "densenet_vs_convnext": dict(zip(
                    ("densenet_only", "convnext_only"),
                    paired_counts(repeat_folds, "single_densenet121", "single_convnext_tiny"),
                )),
            }
            if "convnext_heavy_soft_vote" in summary:
                for strategy in ("four_model_hard_vote", "four_model_equal_soft_vote", "convnext_heavy_soft_vote"):
                    comparisons[str(repeat)][f"{strategy}_vs_convnext"] = dict(zip(
                        ("ensemble_only", "convnext_only"),
                        paired_counts(repeat_folds, strategy, "single_convnext_tiny"),
                    ))
        report[category] = {
            "folds": len(folds),
            "strategies": summary,
            "paired_comparisons_by_repeat": comparisons,
            "paired_comparisons_note": "Discordant counts are reported separately by repeat; repeated predictions are not pooled for significance tests.",
            "validation_note": "One internal early-stopping split is used within each outer training fold. Candidate selection across these CV results remains exploratory and requires an independent evaluation for an unbiased selected-model estimate.",
        }
        print(f"\n{category} ({len(folds)} folds)")
        for name, row in summary.items():
            print(f"  {name:28s} {row['correct']}/{row['total']} acc={row['pooled_accuracy']:.3f} "
                  f"f1={row['pooled_macro_f1']:.3f} repeat_acc="
                  f"{row['repeat_accuracy_mean']:.3f} +/- {row['repeat_accuracy_sd']:.3f} repeat_f1="
                  f"{row['repeat_macro_f1_mean']:.3f} +/- {row['repeat_macro_f1_sd']:.3f}")
        for repeat, comparisons in report[category]["paired_comparisons_by_repeat"].items():
            print(f"  repeat {repeat} paired wins/losses: {comparisons}")
    (input_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
