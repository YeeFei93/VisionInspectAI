from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_classifier(category):
    config = yaml.safe_load((CONFIG_DIR / f"{category}_config.yaml").read_text(encoding="utf-8"))
    return config["defect_classifier"]


def test_category_specific_single_model_choices():
    assert load_classifier("cable")["architecture"] == "efficientnet_b0"
    for category in ("bottle", "capsule", "metal_nut", "wood", "zipper"):
        assert load_classifier(category)["architecture"] == "convnext_tiny"


def test_pill_uses_equal_soft_vote_and_screw_crop_is_preserved():
    pill = load_classifier("pill")
    assert pill["ensemble_architectures"] == [
        "convnext_tiny", "resnet18", "efficientnet_b0", "densenet121"
    ]
    assert "ensemble_weights" not in pill
    assert load_classifier("screw")["use_focused_crops"] is True
    assert load_classifier("screw")["crop_source"] == "patchcore"
