"""
Streamlit demo app for the DIP Team 5 project.

The app lets users upload an image, runs scene classification and diagnostic
fusion, and previews the corrected output side-by-side with the original.

Updated to surface every detail available from the CLI `predict.py` flow:
- Scene classification + confidence + raw label
- AI diagnostic fusion with validation against the manual pipeline
- Difference metrics between preview and validated output
- Recommended filter presets with individual previews
- Optional saving of corrected and preset images to `data/output`

HOW TO RUN : streamlit run app.py --server.port 8501
"""

from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.ai_optimize import generate_preview, load_diagnostic_model, optimize_corrections
from src.ai_scene import classify_image
from src.correction import apply_all_corrections_torch
from src.io_utils import save_image

st.set_page_config(page_title="DIP Color Correction Demo", layout="wide")


# --- Pre/Post-processing helpers (mirrors predict.py) ---
def _preprocess(image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image_np.copy()).to(device)
    return tensor.unsqueeze(0)


def _postprocess(image_tensor: torch.Tensor) -> np.ndarray:
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
    return image_np


def _apply_params(img_rgb: np.ndarray, params: dict, device: torch.device) -> np.ndarray:
    input_tensor = _preprocess(img_rgb, device)

    params_tensor = torch.tensor(
        [
            [
                params["gamma"],
                params["gain_r"],
                params["gain_g"],
                params["gain_b"],
                params["sat"],
                params["hue"],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    corrected_tensor = apply_all_corrections_torch(input_tensor, params_tensor)
    return _postprocess(corrected_tensor[0])


def _validate_application(
    img_rgb: np.ndarray, advice, device: torch.device
) -> tuple[np.ndarray, float, int, float]:
    preview_image_np = generate_preview(img_rgb, advice, device=device)
    validated_image_np = _apply_params(img_rgb, advice.final_params, device)

    diff = np.abs(preview_image_np.astype(np.int16) - validated_image_np.astype(np.int16))
    mean_diff = float(diff.mean())
    max_diff = int(diff.max())

    impact = np.abs(validated_image_np.astype(np.int16) - img_rgb.astype(np.int16))
    mean_impact = float(impact.mean())

    return validated_image_np, mean_diff, max_diff, mean_impact


def _load_diagnostic_model_cached(model_path: Optional[str], device: torch.device):
    """Cache the diagnostic model to avoid reloading on every interaction."""

    @st.cache_resource(show_spinner=False)
    def _load(path: Optional[str], device_str: str):
        resolved_path = Path(path) if path else None
        return load_diagnostic_model(resolved_path, device=torch.device(device_str))

    return _load(model_path, str(device))


def _to_pil_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8))


def _bytes_from_image(image_np: np.ndarray, format: str = "PNG") -> bytes:
    buffer = BytesIO()
    _to_pil_image(image_np).save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


def _build_info_text(scene_info: dict, advice, mean_diff: float, max_diff: int, mean_impact: float) -> str:
    return (
        f"Scene: {scene_info['scene']} (confidence {scene_info['score']:.2f}, raw={scene_info['raw_label']})\n\n"
        "Final Parameters:\n"
        f"  gamma={advice.final_params['gamma']:.3f}\n"
        f"  gains: R={advice.final_params['gain_r']:.3f}, "
        f"G={advice.final_params['gain_g']:.3f}, B={advice.final_params['gain_b']:.3f}\n"
        f"  saturation={advice.final_params['sat']:.3f}, hue={advice.final_params['hue']:.3f}\n\n"
        "Adjustments:\n"
        f"  Exposure: {advice.exposure_adjustment_percent:+.1f}%\n"
        f"  White Balance: {advice.white_balance_shift_percent:.1f}% shift\n"
        f"  Saturation: {advice.saturation_adjustment_percent:+.1f}%\n\n"
        f"Validation: mean diff {mean_diff:.2f}, max diff {max_diff}, mean impact {mean_impact:.2f}\n\n"
        "Summary:\n  " + "\n  ".join(advice.corrections_summary)
    )


def _run_correction(
    img_rgb: np.ndarray,
    mobilenet_path: Optional[str],
    diagnostic_path: Optional[str],
    fusion_weight: float,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene_info = classify_image(img_rgb, weights_path=mobilenet_path)
    diagnostic_model = _load_diagnostic_model_cached(diagnostic_path, device)
    advice = optimize_corrections(
        img_rgb,
        scene_info,
        diagnostic_model=diagnostic_model,
        device=device,
        fusion_weight=fusion_weight,
    )

    corrected_image_np, mean_diff, max_diff, mean_impact = _validate_application(img_rgb, advice, device)

    filter_previews: list[dict] = [{"name": "AI + Preset", "image": corrected_image_np}]
    recommended_previews: list[dict] = []
    if advice.recommended_filters:
        for preset in advice.recommended_filters:
            preset_img = _apply_params(img_rgb, preset["params"], device)
            recommended_previews.append({"name": preset["name"], "image": preset_img})
        filter_previews.extend(recommended_previews)

    info_text = _build_info_text(scene_info, advice, mean_diff, max_diff, mean_impact)

    return {
        "device": device,
        "scene_info": scene_info,
        "advice": advice,
        "corrected_image": corrected_image_np,
        "filter_previews": filter_previews,
        "recommended_previews": recommended_previews,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "mean_impact": mean_impact,
        "info_text": info_text,
    }


def _maybe_save_outputs(result: dict, save_outputs: bool) -> None:
    if not save_outputs:
        return

    output_path = Path("data/output/scene_corrected.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(output_path, result["corrected_image"][..., ::-1])

    if result["recommended_previews"]:
        presets_dir = Path("data/output/presets")
        presets_dir.mkdir(parents=True, exist_ok=True)
        for preset in result["recommended_previews"]:
            preset_path = presets_dir / f"{preset['name'].lower()}_preview.jpg"
            save_image(preset_path, preset["image"][..., ::-1])


# Sidebar controls
st.sidebar.header("Model configuration")
mobilenet_path = st.sidebar.text_input(
    "MobileNetV2 weights path",
    value="weights/mobilenet_v2-b0353104.pth",
    help="Local path to the ImageNet-pretrained MobileNetV2 checkpoint.",
)
diagnostic_path = st.sidebar.text_input(
    "Diagnostic model path",
    value="models/diagnostic_model.pth",
    help="Path to the trained diagnostic CNN checkpoint.",
)
fusion_weight = st.sidebar.slider(
    "AI vs. preset fusion weight",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="0 = use scene presets only, 1 = rely fully on AI diagnostics.",
)
save_outputs = st.sidebar.checkbox(
    "Save corrected + preset previews to data/output", value=False
)


st.title("Color Correction")
with st.container():
    st.info(
        "This system performs **AI-assisted color correction** on uploaded images.\n\n"
        "It combines **scene classification**, **AI diagnostics**, and "
        "**manual DIP algorithms** to automatically enhance exposure, white balance, "
        "and saturation.\n\n"
        "Simply upload a photo to see the improved and recommended filter results."
    )

if "result" not in st.session_state:
    st.session_state["result"] = None
if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
run_clicked = st.button("Run correction", type="primary", width='stretch')

if uploaded is not None and uploaded.name != st.session_state["last_uploaded_name"]:
    st.session_state["result"] = None
    st.session_state["last_uploaded_name"] = uploaded.name

if uploaded:
    input_image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(input_image)

    if st.session_state["result"] is None and not run_clicked:
        st.image(
            input_image,
            caption="Uploaded image",
            use_container_width=True,
        )

    if run_clicked:
        with st.spinner("Running scene analysis and correction..."):
            try:
                result = _run_correction(
                    img_rgb,
                    mobilenet_path or None,
                    diagnostic_path or None,
                    fusion_weight,
                )
            except FileNotFoundError as exc:
                st.error(
                    f"Missing model weights: {exc}. Upload the required files or update the paths."
                )
                st.session_state["result"] = None
            except RuntimeError as exc:
                st.error(
                    f"Model loading failed: {exc}. Ensure weights are available locally."
                )
                st.session_state["result"] = None
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error while processing the image: {exc}")
                st.session_state["result"] = None
            else:
                _maybe_save_outputs(result, save_outputs)
                st.session_state["result"] = result

    result = st.session_state["result"]
    if result is not None:

        if result["max_diff"] > 2 or result["mean_diff"] > 0.5:
            st.warning(
                "Preview path mismatch detected (mean diff > 0.5 or max diff > 2). "
                "Showing validated pipeline output."
            )

        tab_result, tab_detail, tab_filters = st.tabs(
            ["🔍 Result View", "📊 Detailed Analysis", "🎨 Filter Gallery"]
        )

        with tab_result:
            st.markdown(
                f"> ✅ **Scene:** `{result['scene_info']['scene']}`"
            )

            left, right = st.columns(2)
            with left:
                st.subheader("Original")
                st.image(_to_pil_image(img_rgb), use_container_width=True)
            with right:
                name_col, btn_col=st.columns([0.7, 0.3], vertical_alignment='center')
                with name_col:
                    st.subheader("Corrected Image")
                with btn_col:
                    st.download_button(
                        label="Download image",
                        data=_bytes_from_image(result["corrected_image"]),
                        file_name="scene_corrected.png",
                        mime="image/png",
                    )
                st.image(_to_pil_image(result["corrected_image"]), use_container_width=True)
        
        with tab_detail:
            st.subheader("Analysis summary")

            st.markdown(
                f"**Scene:** {result['scene_info']['scene']} "
                f"(confidence {result['scene_info']['score']:.2f}, "
                f"raw label {result['scene_info']['raw_label']})"
            )
            st.markdown(f"**Device:** `{result['device']}`")

            st.markdown(
                "**Final parameters:**  \n"
                f"- gamma = `{result['advice'].final_params['gamma']:.3f}`  \n"
                f"- gains = (`{result['advice'].final_params['gain_r']:.3f}`, "
                f"`{result['advice'].final_params['gain_g']:.3f}`, "
                f"`{result['advice'].final_params['gain_b']:.3f}`)  \n"
                f"- saturation = `{result['advice'].final_params['sat']:.3f}`  \n"
                f"- hue_shift = `{result['advice'].final_params['hue']:.3f}`"
            )

            st.markdown(
                "**Adjustments:**  \n"
                f"- Exposure: `{result['advice'].exposure_adjustment_percent:+.1f}%`  \n"
                f"- White Balance shift: `{result['advice'].white_balance_shift_percent:.1f}%`  \n"
                f"- Saturation: `{result['advice'].saturation_adjustment_percent:+.1f}%`"
            )

            st.markdown(
                "**Validation (preview vs manual pipeline):**  \n"
                f"- mean diff = `{result['mean_diff']:.2f}`  \n"
                f"- max diff = `{result['max_diff']}`  \n"
                f"- mean impact vs. original = `{result['mean_impact']:.2f}`"
            )

        with tab_filters:
            st.subheader("Recommended filter previews")
            if result["filter_previews"]:
                cols = st.columns(2)
                for idx, preview in enumerate(result["filter_previews"]):
                    with cols[idx % 2]:
                        name_col, btn_col = st.columns([0.7,0.3],vertical_alignment='center')
                        with name_col:
                            st.markdown(f"**{preview['name']}**")
                        with btn_col:
                            st.download_button(
                                label=f"Download image",
                                data=_bytes_from_image(preview["image"]),
                                file_name=f"{preview['name'].lower().replace(' ', '_')}.png",
                                mime="image/png",
                                key=f"download-{preview['name']}-{idx}",
                            )
                        st.image(_to_pil_image(preview["image"]), use_container_width=True)
                        
            else:
                st.info("No additional presets were recommended for this image.")

elif run_clicked:
    st.warning("Please upload an image before running the correction.")