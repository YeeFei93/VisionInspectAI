"""VisionInspectAI — Streamlit demo for the screw anomaly detector.

Run:
    streamlit run app/streamlit_app.py

Flow:
    Upload screw image
      -> Prediction: Normal / Defective
      -> Anomaly score
      -> Heatmap overlay (likely defect region highlighted in red)
      -> Defect severity: Low / Medium / High
"""

import json
import sys
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # allow `import src...` when run via `streamlit run`

from src.models.anomaly_detector import PatchCoreAnomalyDetector  # noqa: E402
from src.preprocessing.transform import get_val_transforms  # noqa: E402
from src.visualization.heatmap import make_overlay  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "screw_config.yaml"


@st.cache_resource
def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_detector(_config: dict) -> PatchCoreAnomalyDetector:
    anomaly_cfg = _config["anomaly_detection"]
    category = _config.get("category", "screw")
    run_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    checkpoint_path = PROJECT_ROOT / _config["output"]["checkpoint_dir"] / f"{run_name}_memory_bank.pt"

    if not checkpoint_path.exists():
        st.error(
            f"No trained memory bank found at {checkpoint_path}. "
            "Run `python -m src.models.run_anomaly_detection` first."
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
def load_metrics(_config: dict) -> dict:
    anomaly_cfg = _config["anomaly_detection"]
    category = _config.get("category", "screw")
    run_name = f"patchcore_{anomaly_cfg['backbone']}_{category}"
    metrics_path = PROJECT_ROOT / _config["output"]["metrics_dir"] / f"{run_name}_metrics.json"

    if not metrics_path.exists():
        st.error(
            f"No metrics file found at {metrics_path}. "
            "Run `python -m src.models.run_anomaly_detection` first."
        )
        st.stop()

    with metrics_path.open() as f:
        return json.load(f)


def classify_severity(score: float, threshold: float, score_max: float):
    """Bucket a defective anomaly score into Low / Medium / High severity,
    scaled by how far the score sits between the decision threshold and the
    highest score observed during evaluation."""
    if score < threshold:
        return None
    span = max(score_max - threshold, 1e-6)
    normalized = min(max((score - threshold) / span, 0.0), 1.0)
    if normalized < 1 / 3:
        return "Low"
    if normalized < 2 / 3:
        return "Medium"
    return "High"


def main() -> None:
    st.set_page_config(page_title="VisionInspectAI — Screw Inspection", layout="wide")
    st.title("VisionInspectAI — Screw Anomaly Detection")
    st.caption(
        "Upload a screw image to check whether it is normal or defective, "
        "and see the suspected defect region."
    )

    config = load_config()
    detector = load_detector(config)
    metrics = load_metrics(config)

    threshold = metrics["threshold"]
    score_max = metrics.get("score_max", threshold * 1.5)
    image_size = config["data"]["image_size"]
    transform = get_val_transforms(image_size)

    uploaded_file = st.file_uploader("Upload a screw image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Upload a .png / .jpg screw image to run inspection.")
        return

    original_image = Image.open(uploaded_file).convert("RGB")
    resized_image = original_image.resize((image_size, image_size))
    input_tensor = transform(original_image).unsqueeze(0)

    with st.spinner("Running anomaly detection..."):
        result = detector.predict(input_tensor)[0]

    score = result.image_score
    prediction = "Defective" if score >= threshold else "Normal"
    severity = classify_severity(score, threshold, score_max)

    _normalized_map, heatmap_rgb, overlay = make_overlay(resized_image, result.anomaly_map)

    image_col, heatmap_col, overlay_col = st.columns(3)
    image_col.image(resized_image, caption="Original", use_container_width=True)
    heatmap_col.image(heatmap_rgb, caption="Anomaly heatmap", use_container_width=True)
    overlay_col.image(overlay, caption="Overlay (likely defect region in red)", use_container_width=True)

    st.subheader("Result")
    prediction_col, score_col, severity_col = st.columns(3)
    prediction_col.metric("Prediction", prediction)
    score_col.metric("Anomaly score", f"{score:.2f}")
    severity_col.metric("Severity", severity or "—")

    if prediction == "Defective":
        st.error(
            f"Prediction: {prediction}\n\n"
            f"Anomaly score: {score:.2f}\n\n"
            f"Severity: {severity}\n\n"
            "Likely defect region: highlighted in red"
        )
    else:
        st.success(f"Prediction: {prediction}\n\nAnomaly score: {score:.2f}")


if __name__ == "__main__":
    main()
