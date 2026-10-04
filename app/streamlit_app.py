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

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # allow `import src...` when run via `streamlit run`

from src.inference.inspection_pipeline import (  # noqa: E402
    InspectionPipeline,
    InspectionSetupError,
    LowCategoryConfidenceError,
)
from src.visualization.heatmap import create_defect_type_overlay  # noqa: E402


def load_as_rgb(uploaded_file) -> Image.Image:
    """Open an uploaded image and flatten it to RGB.

    Many web/stock photos are RGBA PNGs with a transparent background.
    Naively calling ``.convert("RGB")`` on those keeps whatever arbitrary
    (often black or garbage) colour value sits behind the transparent
    pixels, which PatchCore/the classifiers then treat as real texture —
    inflating the anomaly score for reasons that have nothing to do with
    the object itself. Compositing onto a neutral grey background first
    avoids that artifact and is closer to MVTec-AD's plain backgrounds.
    """
    image = Image.open(uploaded_file)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (128, 128, 128))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


@st.cache_resource
def load_inspection_pipeline() -> InspectionPipeline:
    return InspectionPipeline(project_root=PROJECT_ROOT)


def main() -> None:
    st.set_page_config(page_title="VisionInspectAI", layout="wide")
    st.title("VisionInspectAI — MVTec-AD Anomaly Detection")
    st.caption(
        "Upload an image — the object type is detected automatically, then it's checked for "
        "defects and the suspected defect region is shown."
    )

    pipeline = load_inspection_pipeline()
    try:
        known_categories = pipeline.known_categories
    except InspectionSetupError as error:
        st.error(str(error))
        return

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info(
            "Upload a .png / .jpg image of one of the supported categories: "
            f"{', '.join(sorted(known_categories))}."
        )
        return

    original_image = load_as_rgb(uploaded_file)

    with st.expander("Override automatic category detection"):
        automatic_option = "Auto-detect"
        category_option = st.selectbox(
            "Category", [automatic_option, *sorted(known_categories)]
        )
    category = None if category_option == automatic_option else category_option

    try:
        with st.spinner("Running anomaly detection..."):
            result = pipeline.inspect(original_image, category=category)
    except LowCategoryConfidenceError as error:
        st.write(
            f"**Detected category:** {error.category} "
            f"(confidence {error.confidence:.0%})"
        )
        st.warning(
            "Category not found. Confidence is below 60%, so no prediction is shown for this image. "
            "Please try another image or choose a category manually to continue."
        )
        return
    except InspectionSetupError as error:
        st.error(str(error))
        return

    st.write(
        f"**Detected category:** {result.detected_category} "
        f"(confidence {result.category_confidence:.0%})"
    )
    if result.category_confidence < 0.7:
        st.warning(
            "Low detection confidence — this image may not resemble the trained categories closely "
            "enough (different framing/background/lighting than MVTec-AD, or not one of the trained "
            "object types at all). Verify the selected category before trusting the result."
        )
    if result.category != result.detected_category:
        st.caption(f"Using manually selected category: **{result.category}**")

    st.subheader("Image Analysis")
    
    # Determine if we should show the color-coded defect type overlay
    show_defect_types_overlay = result.prediction == "Defective" and len(result.distinct_defect_regions) > 0
    image_size = result.resized_image.size[0]
    
    if show_defect_types_overlay:
        # 4-column layout: Original | Heatmap | Defect Types | Overlay
        image_col, heatmap_col, defect_col, overlay_col = st.columns(4)
        
        # Create the color-coded defect type overlay
        defect_colored_mask, defect_type_overlay, type_to_color = create_defect_type_overlay(
            result.resized_image, result.distinct_defect_regions, image_size, alpha=0.5
        )
        
        image_col.image(result.resized_image, caption="Original", use_container_width=True)
        heatmap_col.image(
            result.heatmap,
            caption="Anomaly heatmap",
            use_container_width=True,
        )
        defect_col.image(
            defect_type_overlay,
            caption="Defect type overlay (color-coded)",
            use_container_width=True,
        )
        overlay_col.image(result.overlay, caption="Heatmap overlay", use_container_width=True)
    else:
        # 3-column layout: Original | Heatmap | Overlay
        image_col, heatmap_col, overlay_col = st.columns(3)
        image_col.image(result.resized_image, caption="Original", use_container_width=True)
        heatmap_col.image(
            result.heatmap,
            caption="PatchCore response (yellow-red = primary predicted region)",
            use_container_width=True,
        )
        overlay_col.image(
            result.overlay,
            caption="Primary predicted region (approximate, not pixel-exact)",
            use_container_width=True,
        )

    st.subheader("Result")
    prediction_col, score_col, severity_col = st.columns(3)
    prediction_col.metric("Prediction", result.prediction)
    score_col.metric(
        "Anomaly score",
        f"{result.anomaly_score:.2f}",
        delta=f"threshold {result.threshold:.2f}",
        delta_color="off",
    )
    severity_col.metric("Severity", result.severity or "—")

    if result.prediction == "Defective":
        if result.defect_types:
            primary_defect = result.defect_types[0]
            st.metric(
                "Likely defect type",
                primary_defect.defect_type,
                delta=f"confidence {primary_defect.confidence:.0%}",
                delta_color="off",
            )
        elif result.defect_model_available:
            st.caption(
                "Defect-type classifier ran but its prediction confidence was below 60%, so no defect type is shown."
            )
        else:
            st.caption(
                f"No defect-type classifier trained for **{category}** yet — run "
                f"`python -m src.models.train_defect_classifier --config config/{category}_config.yaml` to enable this."
            )

        if len(result.distinct_defect_regions) > 1:
            st.write("**Multiple defect types detected in this image (region-based, approximate):**")
            for region in result.distinct_defect_regions:
                st.write(f"- {region.defect_type} — confidence {region.confidence:.0%}, region area {region.area}px")
            st.caption(
                "Each anomalous region (connected component of the heatmap above the threshold) is cropped "
                "and classified independently. The color-coded defect type overlay above visually shows which "
                "color represents each defect type. MVTec-AD's labels don't break a 'combined' image down into "
                "individual defect types, so this is an approximate visual breakdown."
            )
        st.error(f"{result.severity_reason} Review the red overlay for the likely defect region.")
    else:
        st.success("No anomalous region was detected.")


if __name__ == "__main__":
    main()
