import pandas as pd
import pytest
import torch
from torch import nn

import src.models.train_baseline as train_baseline


def test_early_stopping_restores_best_epoch_weights(monkeypatch):
    model = nn.Linear(1, 2, bias=False)
    model.weight.data.zero_()
    epoch = 0
    val_losses = iter([0.5, 0.3, 0.4, 0.5, 0.3])

    def fake_train_one_epoch(model, _loader, _criterion, _optimizer, _device):
        nonlocal epoch
        epoch += 1
        model.weight.data.fill_(epoch)
        return float(epoch), 1.0

    def fake_evaluate(_model, _loader, _criterion, _device):
        return next(val_losses), 1.0, [0, 1], [0, 1]

    monkeypatch.setattr(train_baseline, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_baseline, "evaluate", fake_evaluate)

    _y_true, _y_pred, _val_loss, summary = train_baseline.train_with_early_stopping(
        model,
        train_loader=object(),
        val_loader=object(),
        train_cfg={
            "epochs": 10,
            "learning_rate": 0.0,
            "early_stopping_patience": 2,
        },
        device=torch.device("cpu"),
    )

    assert summary["best_epoch"] == 2
    assert summary["epochs_trained"] == 4
    assert torch.all(model.weight == 2)


def test_cross_validation_rejects_more_folds_than_smallest_class():
    manifest = pd.DataFrame(
        {
            "split": ["test"] * 5,
            "label": [0, 0, 1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="smallest class count"):
        train_baseline.run_cross_validation(
            manifest,
            data_cfg={"seed": 42},
            model_cfg={},
            train_cfg={},
            device=torch.device("cpu"),
            folds=3,
        )