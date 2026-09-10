import pytest

from src.models.baseline_classifier import (
    apply_freeze_mode,
    build_baseline_model,
    freeze_backbone,
    unfreeze_last_block,
)


def _trainable_param_names(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_freeze_backbone_only_trains_head():
    model = build_baseline_model("resnet18", num_classes=2, pretrained=False)
    freeze_backbone(model, "resnet18")

    trainable = _trainable_param_names(model)
    assert trainable == {name for name, _ in model.fc.named_parameters(prefix="fc")}
    assert all(name.startswith("fc.") for name in trainable)


def test_unfreeze_last_block_trains_head_and_last_block_only():
    model = build_baseline_model("resnet18", num_classes=2, pretrained=False)
    unfreeze_last_block(model, "resnet18")

    trainable = _trainable_param_names(model)
    assert trainable
    assert all(name.startswith(("fc.", "layer4.")) for name in trainable)
    # layer1-3 must stay frozen, confirming this is stricter than full fine-tuning.
    assert not any(name.startswith(("layer1.", "layer2.", "layer3.")) for name in trainable)


def test_apply_freeze_mode_none_leaves_everything_trainable():
    model = build_baseline_model("resnet18", num_classes=2, pretrained=False)
    apply_freeze_mode(model, "resnet18", "none")

    assert all(param.requires_grad for param in model.parameters())


def test_apply_freeze_mode_rejects_unknown_mode():
    model = build_baseline_model("resnet18", num_classes=2, pretrained=False)

    with pytest.raises(ValueError, match="Unsupported freeze_mode"):
        apply_freeze_mode(model, "resnet18", "bogus")
