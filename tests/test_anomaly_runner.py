from argparse import Namespace

from src.models.run_anomaly_detection import build_run_name


def tuning_args(**overrides):
    values = {
        "seed": None,
        "max_coreset_size": None,
        "projection_dim": None,
        "layers": None,
        "num_neighbors": None,
        "softmax_reweighting": None,
        "reweight_num_neighbors": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_run_name_keeps_default_name_compact():
    name = build_run_name("resnet18", "screw", "random", 42, tuning_args())

    assert name == "patchcore_resnet18_screw"


def test_build_run_name_identifies_every_tuned_parameter():
    args = tuning_args(
        seed=41,
        max_coreset_size=1000,
        projection_dim=64,
        layers=["layer1", "layer2"],
    )

    name = build_run_name("resnet18", "screw", "pca", 41, args)

    assert name == "patchcore_resnet18_screw_pcaproj_seed41_cs1000_pd64_layer1-layer2"


def test_build_run_name_can_include_config_defined_seed():
    name = build_run_name(
        "resnet18", "screw", "pca", 42, tuning_args(), include_seed=True
    )

    assert name == "patchcore_resnet18_screw_pcaproj_seed42"


def test_build_run_name_identifies_config_defined_tuning():
    name = build_run_name(
        "resnet18",
        "screw",
        "random",
        42,
        tuning_args(),
        max_coreset_size=4000,
        projection_dim=256,
        layers=("layer2",),
    )

    assert name == "patchcore_resnet18_screw_cs4000_pd256_layer2"


def test_build_run_name_identifies_scoring_options():
    name = build_run_name(
        "resnet18",
        "screw",
        "random",
        42,
        tuning_args(),
        num_neighbors=3,
        softmax_reweighting=True,
        reweight_num_neighbors=5,
    )

    assert name == "patchcore_resnet18_screw_knn3_rw5"