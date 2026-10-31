import json
from pathlib import Path

from src.inference.inspection_pipeline import DEFAULT_CATEGORY_CONFIGS, PROJECT_ROOT


def test_every_routable_category_has_a_config_file():
    for category, path in DEFAULT_CATEGORY_CONFIGS.items():
        assert path.exists(), f"missing config for {category}"


def test_every_category_the_classifier_can_predict_is_routable():
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "category_classifier_metrics.json"
    if not metrics_path.exists():
        return  # trained artifacts are gitignored; nothing to cross-check on a fresh clone
    categories = json.loads(Path(metrics_path).read_text())["categories"]
    assert set(categories) <= set(DEFAULT_CATEGORY_CONFIGS)
