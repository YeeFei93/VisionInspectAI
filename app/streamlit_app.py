"""VisionInspectAI — Streamlit demo for the MVTec-AD anomaly detector.

Run:
    streamlit run app/streamlit_app.py

Flow:
    Upload an image
      -> Object/category auto-detected (screw / bottle / hazelnut / ...)
      -> Prediction: Normal / Defective
      -> Anomaly score
      -> Heatmap overlay (likely defect region highlighted in red)
      -> Defect severity: Low / Medium / High
      -> Defect type (e.g. leather: color / cut / fold / glue / poke), if defective
         and a defect-type classifier has been trained for the category
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # allow `import src...` when run via `streamlit run`

from src.models.anomaly_detector import PatchCoreAnomalyDetector  # noqa: E402
from src.models.baseline_classifier import build_baseline_model  # noqa: E402
from src.preprocessing.segmentation import compute_foreground_mask  # noqa: E402
from src.preprocessing.transform import get_val_transforms  # noqa: E402
from src.visualization.heatmap import make_overlay  # noqa: E402

# Category -> its primary config file. Add an entry here after running
# create_manifest.py + train_baseline.py + run_anomaly_detection.py for a
# new MVTec-AD category, and retrain the category classifier
# (src/models/train_category_classifier.py) so it can recognize it too.
CATEGORY_CONFIGS = {
    "screw": PROJECT_ROOT / "config" / "screw_config.yaml",
    "bottle": PROJECT_ROOT / "config" / "bottle_config.yaml",
    "hazelnut": PROJECT_ROOT / "config" / "hazelnut_config.yaml",
    "carpet": PROJECT_ROOT / "config" / "carpet_config.yaml",
    "leather": PROJECT_ROOT / "config" / "leather_config.yaml",
}

CATEGORY_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "category_classifier_resnet18.pt"
CATEGORY_CLASSIFIER_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "category_classifier_metrics.json"
CATEGORY_CLASSIFIER_IMAGE_SIZE = 224


@st.cache_resource
def load_category_classifier():
    """Load the "which object is this?" classifier (see
    src/models/train_category_classifier.py), used to auto-detect the
    category so the user doesn't have to pick one manually."""
    if not CATEGORY_CLASSIFIER_CHECKPOINT.exists() or not CATEGORY_CLASSIFIER_METRICS.exists():
        st.error(
            "No category classifier found. Run "
            "`python -m src.models.train_category_classifier` first."
        )
        st.stop()

    with CATEGORY_CLASSIFIER_METRICS.open() as f:
        categories = json.load(f)["categories"]

    model = build_baseline_model(architecture="resnet18", num_classes=len(categories), pretrained=False)
    model.load_state_dict(torch.load(CATEGORY_CLASSIFIER_CHECKPOINT, map_location="cpu"))
    model.eval()
    return model, categories


def detect_category(model, categories, original_image: Image.Image):
    transform = get_val_transforms(CATEGORY_CLASSIFIER_IMAGE_SIZE)
    input_tensor = transform(original_image).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    return categories[pred_idx], float(probs[pred_idx].item())


@st.cache_resource
def load_config(category: str) -> dict:
    with CATEGORY_CONFIGS[category].open() as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_detector(category: str, _config: dict) -> PatchCoreAnomalyDetector:
    anomaly_cfg = _config["anomaly_detection"]
    run_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    checkpoint_path = PROJECT_ROOT / _config["output"]["checkpoint_dir"] / f"{run_name}_memory_bank.pt"

    if not checkpoint_path.exists():
        st.error(
            f"No trained memory bank found at {checkpoint_path}. "
            f"Run `python -m src.models.run_anomaly_detection --config config/{category}_config.yaml` first."
        )
        st.stop()

    detector = PatchCoreAnomalyDetector(
        backbone=anomaly_cfg["backbone"],
        layers=tuple(anomaly_cfg["layers"]),
        device="cpu",
    )
    detector.load(checkpoint_path)
    return detector


@st.cache_resource
def load_metrics(category: str, _config: dict) -> dict:
    anomaly_cfg = _config["anomaly_detection"]
    run_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    metrics_path = PROJECT_ROOT / _config["output"]["metrics_dir"] / f"{run_name}_metrics.json"

    if not metrics_path.exists():
        st.error(
            f"No metrics file found at {metrics_path}. "
            f"Run `python -m src.models.run_anomaly_detection --config config/{category}_config.yaml` first."
        )
        st.stop()

    with metrics_path.open() as f:
        return json.load(f)


@st.cache_resource
def load_defect_classifier(category: str, _config: dict):
    """Load the per-category "what kind of defect is this?" classifier (see
    src/models/train_defect_classifier.py). Returns (model, defect_types), or
    (None, None) if it hasn't been trained yet for this category — the
    defect-type breakdown is an optional add-on, not required for the
    Normal/Defective verdict."""
    model_cfg = _config["model"]
    run_name = f"defect_classifier_{model_cfg['architecture']}_{category}"
    checkpoint_path = PROJECT_ROOT / _config["output"]["checkpoint_dir"] / f"{run_name}.pt"
    metrics_path = PROJECT_ROOT / _config["output"]["metrics_dir"] / f"{run_name}_metrics.json"

    if not checkpoint_path.exists() or not metrics_path.exists():
        return None, None

    with metrics_path.open() as f:
        defect_types = json.load(f)["defect_types"]

    model = build_baseline_model(
        architecture=model_cfg["architecture"], num_classes=len(defect_types), pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model, defect_types


def detect_defect_type(model, defect_types, input_tensor: torch.Tensor):
    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    return defect_types[pred_idx], float(probs[pred_idx].item()), probs.tolist()


def classify_severity(anomaly_map: torch.Tensor, foreground_mask: np.ndarray, threshold: float):
    """Bucket a defective prediction into Low / Medium / High severity based
    on how much of the object's surface area is anomalous (not just the raw
    score), with a short human-readable reason."""
    anomaly_map_np = anomaly_map.detach().cpu().numpy()
    foreground_area = int(foreground_mask.sum())
    if foreground_area == 0:
        return None, "No object detected in the image."

    anomalous_area = int(np.logical_and(anomaly_map_np >= threshold, foreground_mask).sum())
    fraction = anomalous_area / foreground_area

    if fraction < 0.05:
        return "Low", "Anomaly area is small and localized."
    if fraction < 0.20:
        return "Medium", "Anomaly covers a moderate portion of the object."
    return "High", "Anomaly covers a large portion of the object."


def main() -> None:
    st.set_page_config(page_title="VisionInspectAI", layout="wide")
    st.title("VisionInspectAI — MVTec-AD Anomaly Detection")
    st.caption(
        "Upload an image — the object type is detected automatically, then it's checked for "
        "defects and the suspected defect region is shown."
    )

    category_model, known_categories = load_category_classifier()

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info(f"Upload a .png / .jpg image of one of: {', '.join(sorted(known_categories))}.")
        return

    original_image = Image.open(uploaded_file).convert("RGB")

    detected_category, confidence = detect_category(category_model, known_categories, original_image)
    st.write(f"**Detected category:** {detected_category} (confidence {confidence:.0%})")

    with st.expander("Not correct? Override the detected category"):
        options = sorted(known_categories)
        category = st.selectbox("Category", options, index=options.index(detected_category))
    if category != detected_category:
        st.caption(f"Using manually selected category: **{category}**")

    config = load_config(category)
    detector = load_detector(category, config)
    metrics = load_metrics(category, config)

    threshold = metrics["threshold"]
    image_size = config["data"]["image_size"]
    use_foreground_mask = config["anomaly_detection"].get("use_foreground_mask", True)
    transform = get_val_transforms(image_size)

    resized_image = original_image.resize((image_size, image_size))
    input_tensor = transform(original_image).unsqueeze(0)

    if use_foreground_mask:
        foreground_mask = compute_foreground_mask(resized_image, image_size)
        foreground_mask_tensor = torch.from_numpy(foreground_mask).unsqueeze(0)
    else:
        # No plain background to mask out for this category (object fills the
        # whole frame) — treat the entire image as the object of interest.
        foreground_mask = np.ones((image_size, image_size), dtype=bool)
        foreground_mask_tensor = None

    with st.spinner("Running anomaly detection..."):
        result = detector.predict(input_tensor, foreground_masks=foreground_mask_tensor)[0]

    score = result.image_score
    prediction = "Defective" if score >= threshold else "Normal"
    severity, severity_reason = classify_severity(result.anomaly_map, foreground_mask, threshold)

    defect_type, defect_confidence = None, None
    if prediction == "Defective":
        defect_model, defect_types = load_defect_classifier(category, config)
        if defect_model is not None:
            defect_type, defect_confidence, _ = detect_defect_type(defect_model, defect_types, input_tensor)

    _normalized_map, heatmap_rgb, overlay = make_overlay(resized_image, result.anomaly_map, threshold=threshold)

    image_col, heatmap_col, overlay_col = st.columns(3)
    image_col.image(resized_image, caption="Original", use_container_width=True)
    heatmap_col.image(
        heatmap_rgb,
        caption="Anomaly heatmap (red = above decision threshold)",
        use_container_width=True,
    )
    overlay_col.image(overlay, caption="Overlay (likely defect region in red)", use_container_width=True)

    st.subheader("Result")
    prediction_col, score_col, severity_col = st.columns(3)
    prediction_col.metric("Prediction", prediction)
    score_col.metric("Anomaly score", f"{score:.2f}", delta=f"threshold {threshold:.2f}", delta_color="off")
    severity_col.metric("Severity", severity or "—")

    if prediction == "Defective":
        if defect_type is not None:
            st.metric("Likely defect type", defect_type, delta=f"confidence {defect_confidence:.0%}", delta_color="off")
        else:
            st.caption(
                f"No defect-type classifier trained for **{category}** yet — run "
                f"`python -m src.models.train_defect_classifier --config config/{category}_config.yaml` to enable this."
            )
        st.error(
            f"Prediction: {prediction}\n\n"
            f"Anomaly score: {score:.2f} (decision threshold: {threshold:.2f})\n\n"
            + (f"Defect type: {defect_type} (confidence {defect_confidence:.0%})\n\n" if defect_type else "")
            + f"Severity: {severity} — {severity_reason}\n\n"
            "Likely defect region: highlighted in red"
        )
    else:
        st.success(
            f"Prediction: {prediction}\n\n"
            f"Anomaly score: {score:.2f} (decision threshold: {threshold:.2f})"
        )


if __name__ == "__main__":
    main()
