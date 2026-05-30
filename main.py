"""
Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    or
    venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
"""

import io
import base64
import warnings
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import tensorflow as tf
import keras

try:
    tf.config.set_visible_devices([], "GPU")
    DEVICE_MODE = "CPU"
except Exception as exc:
    DEVICE_MODE = f"default ({exc})"

from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# Configuration
IMG_SIZE      = (224, 224)
THRESHOLD     = 0.35
UNCERTAIN_LOW = 0.30
UNCERTAIN_HIGH = 0.55
MAX_TTA_PASSES = 8
DEFAULT_TTA_PASSES = 5
DISPLAY_MAX_SIZE = 900
ALPHA_OVERLAY = 0.42
MODEL_PATH    = Path("model/resnet50_mura_finetuned.keras")
MODEL_FAMILY  = "resnet50"
MODEL_DISPLAY_NAME = "ResNet50_MURA"

Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app = FastAPI(title="X-Ray Fracture Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

model               = None
preprocess_fn       = None
backbone_feat_model = None
head_layers         = []
conv_layer_name     = None


# Startup
def get_preprocess_fn(family: str):
    return efficientnet_preprocess if "efficient" in family.lower() else resnet_preprocess


@app.on_event("startup")
def load_model():
    global model, preprocess_fn, backbone_feat_model, head_layers, conv_layer_name

    preprocess_fn = get_preprocess_fn(MODEL_FAMILY)

    if not MODEL_PATH.exists():
        print(f"WARNING: model not found at '{MODEL_PATH}' — running in DEMO MODE.")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f"  Loaded: {model.name}")

    # Find nested backbone
    backbone = None
    backbone_idx = None
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "layers") and "resnet" in layer.name.lower():
            backbone = layer
            backbone_idx = i
            break

    # Find last 4D conv layer inside backbone
    last_conv = None
    for layer in reversed(backbone.layers):
        try:
            out = layer.output
            if isinstance(out, list):
                out = out[0]
            if len(out.shape) == 4:
                last_conv = layer
                conv_layer_name = layer.name
                break
        except Exception:
            continue

    # Get the correct output tensors (backbone.output is a list with 1 element)
    last_conv_out      = last_conv.output
    if isinstance(last_conv_out, list):
        last_conv_out  = last_conv_out[0]

    backbone_final_out = backbone.output
    if isinstance(backbone_final_out, list):
        backbone_final_out = backbone_final_out[-1]

    backbone_feat_model = keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_out, backbone_final_out],
        name="backbone_feat_model"
    )

    head_layers = model.layers[backbone_idx + 1:]
    print(f"  Conv layer : {conv_layer_name}")
    print(f"  Head layers: {[l.name for l in head_layers]}")
    print("  Ready.")


# Preprocessing & decision helpers
def preprocess_array(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32).copy()
    return preprocess_fn(arr)


def make_tta_variants(original_rgb: np.ndarray, passes: int) -> list[np.ndarray]:
    passes = max(1, min(int(passes), MAX_TTA_PASSES))
    h, w = original_rgb.shape[:2]
    center = (w / 2, h / 2)

    def rotate(deg: float) -> np.ndarray:
        matrix = cv2.getRotationMatrix2D(center, deg, 1.0)
        return cv2.warpAffine(
            original_rgb, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def brightness(scale: float) -> np.ndarray:
        return np.clip(original_rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    variants = [
        original_rgb,
        cv2.flip(original_rgb, 1),
        rotate(5),
        rotate(-5),
        brightness(0.92),
        brightness(1.08),
        np.roll(original_rgb, shift=4, axis=1),
        np.roll(original_rgb, shift=-4, axis=0),
    ]
    return variants[:passes]


def display_image_array(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    img.thumbnail((DISPLAY_MAX_SIZE, DISPLAY_MAX_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def preprocess_image(pil_img: Image.Image, tta_passes: int = 1):
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    model_rgb = np.array(img, dtype=np.uint8)
    display_rgb = display_image_array(pil_img)
    variants = make_tta_variants(model_rgb, tta_passes)
    batch = np.stack([preprocess_array(v) for v in variants]).astype(np.float32)
    return display_rgb, batch


def classify_probability(prob: float) -> dict:
    if prob >= UNCERTAIN_HIGH:
        return {
            "label": "Fracture Suspected",
            "label_type": "fracture",
            "risk_level": "high",
            "gradcam_target": "fracture",
            "recommendation": "Radiologist review recommended. Warm Grad-CAM areas explain fracture-suspicion evidence.",
        }
    if prob <= UNCERTAIN_LOW:
        return {
            "label": "No Fracture",
            "label_type": "normal",
            "risk_level": "low",
            "gradcam_target": "normal",
            "recommendation": "Model leans normal. This does not replace professional diagnosis.",
        }
    return {
        "label": "Review Recommended",
        "label_type": "uncertain",
        "risk_level": "uncertain",
        "gradcam_target": "fracture" if prob >= 0.5 else "normal",
        "recommendation": "Score is close to the decision boundary; manual clinical review is recommended.",
    }


def analyze_image_quality(pil_img: Image.Image) -> dict:
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    h, w = gray.shape[:2]
    mean = float(gray.mean())
    contrast = float(gray.std())
    dynamic_range = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    dark_ratio = float(np.mean(gray < 8))
    bright_ratio = float(np.mean(gray > 247))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    saturation_mean = float(saturation.mean())
    saturated_pixel_ratio = float(np.mean(saturation > 45))

    channel_spread = (
        rgb.astype(np.float32).max(axis=2) - rgb.astype(np.float32).min(axis=2)
    )
    color_spread_mean = float(channel_spread.mean())

    edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
    edge_density = float(np.mean(edges > 0))

    warnings_out = []

    if min(w, h) < 160:
        warnings_out.append("Low resolution image")
    if mean < 20:
        warnings_out.append("Very dark image")
    elif mean > 235:
        warnings_out.append("Very bright image")
    if contrast < 18:
        warnings_out.append("Low contrast image")

    return {
        "status": "warning" if warnings_out else "ok",
        "width": int(w),
        "height": int(h),
        "mean_intensity": round(mean, 1),
        "contrast": round(contrast, 1),
        "dynamic_range": round(dynamic_range, 1),
        "dark_pixel_ratio": round(dark_ratio, 3),
        "bright_pixel_ratio": round(bright_ratio, 3),
        "saturation_mean": round(saturation_mean, 1),
        "saturated_pixel_ratio": round(saturated_pixel_ratio, 3),
        "color_spread_mean": round(color_spread_mean, 1),
        "edge_density": round(edge_density, 3),
        "warnings": warnings_out,
    }


def validate_xray_input(pil_img: Image.Image) -> dict:
    """
    Lightweight pre-validation for obviously invalid or non-radiographic images.
    This is a heuristic guard, not a substitute for a trained xray/not-xray model.
    """
    quality = analyze_image_quality(pil_img)
    reasons = []

    if min(quality["width"], quality["height"]) < 96:
        reasons.append("Image resolution is too low for analysis.")

    nearly_blank = (
        quality["contrast"] < 3.0
        or quality["dynamic_range"] < 10.0
        or quality["dark_pixel_ratio"] > 0.98
        or quality["bright_pixel_ratio"] > 0.98
    )
    if nearly_blank:
        reasons.append("The image is almost blank, fully black/white, or has too little contrast.")

    bright_document_like = (
        quality["mean_intensity"] > 180.0
        and quality["bright_pixel_ratio"] > 0.35
        and quality["dark_pixel_ratio"] < 0.08
    )
    if bright_document_like:
        reasons.append("The image looks like a bright document, screenshot, or diagram rather than an X-ray scan.")

    strongly_colored = (
        quality["saturation_mean"] > 12.0
        and quality["saturated_pixel_ratio"] > 0.03
        and quality["color_spread_mean"] > 6.0
    )
    if strongly_colored:
        reasons.append("The image contains too much color for a typical X-ray scan.")

    if quality["edge_density"] < 0.002 and quality["contrast"] < 12.0:
        reasons.append("The image does not contain enough visible anatomical structure.")

    return {
        "is_valid": not reasons,
        "quality": quality,
        "reasons": reasons,
    }


# Grad-CAM
def make_gradcam_heatmap(batch: np.ndarray, target_class: str = "fracture") -> np.ndarray:
    """
    Compute a classic Grad-CAM heatmap for the requested binary class.

    For a sigmoid binary model, x[:, 0] is evidence for the abnormal/fracture
    class. The normal class can be interpreted as 1 - x[:, 0].
    """
    img_tensor = tf.cast(batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_out = backbone_feat_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        x = backbone_out
        for layer in head_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)

        prob_fracture = x[:, 0]
        class_score = 1.0 - prob_fracture if target_class == "normal" else prob_fracture

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        print("WARNING: gradients are None.")
        h, w = conv_outputs.shape[1], conv_outputs.shape[2]
        return np.ones((h, w), dtype=np.float32) * 0.5

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_out     = conv_outputs[0]                          # (7, 7, 2048)
    heatmap      = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (7, 7)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / (max_val + 1e-8), heatmap)

    return heatmap.numpy()


def build_overlay(original_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w        = original_rgb.shape[:2]
    hm          = cv2.resize(heatmap, (w, h))
    hm_uint8    = np.uint8(255 * hm)
    colored_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    alpha       = (hm[..., None] * ALPHA_OVERLAY).astype(np.float32)
    return np.uint8((1 - alpha) * original_rgb + alpha * colored_rgb)


def heatmap_to_rgba_b64(original_rgb: np.ndarray, heatmap: np.ndarray) -> str:
    h, w = original_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_uint8 = np.uint8(255 * hm)
    colored_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([colored_rgb, hm_uint8])

    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# Helpers
def to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def demo_predict(pil_img: Image.Image) -> dict:
    import random
    prob  = round(random.uniform(0.15, 0.85), 4)
    decision = classify_probability(prob)
    conf  = prob if prob > 0.5 else 1.0 - prob
    heatmap = np.zeros((7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            heatmap[i, j] = np.exp(-((i-3)**2 + (j-3)**2) / 4.0)
    original_rgb = display_image_array(pil_img)
    return {
        "label": decision["label"],
        "label_type": decision["label_type"],
        "risk_level": decision["risk_level"],
        "recommendation": decision["recommendation"],
        "prob_fracture": prob,
        "prob_normal": round(1.0 - prob, 4),
        "confidence": round(conf * 100, 1),
        "threshold": THRESHOLD,
        "decision_thresholds": {
            "normal_max": UNCERTAIN_LOW,
            "review_range": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
            "fracture_min": UNCERTAIN_HIGH,
            "research_threshold": THRESHOLD,
        },
        "prediction_std": 0.0,
        "tta_passes": DEFAULT_TTA_PASSES,
        "model_name": MODEL_DISPLAY_NAME,
        "device_mode": DEVICE_MODE,
        "demo_mode": True,
        "is_uncertain": decision["label_type"] == "uncertain",
        "gradcam_target": decision["gradcam_target"],
        "quality": analyze_image_quality(pil_img),
        "original_b64": to_b64(original_rgb),
        "heatmap_b64": heatmap_to_rgba_b64(original_rgb, heatmap),
        "gradcam_b64":  to_b64(build_overlay(original_rgb, heatmap)),
    }


# Routes
@app.get("/", response_class=HTMLResponse)
def root():
    p = Path("templates/index.html")
    return HTMLResponse(content=p.read_text(encoding="utf-8") if p.exists()
                        else "<h1>templates/index.html not found</h1>")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tta_passes: int = Query(DEFAULT_TTA_PASSES, ge=1, le=MAX_TTA_PASSES),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(400, "Unsupported file type. Upload JPG, PNG, or BMP.")
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large — max 20 MB.")
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Cannot open image file.")

    validation = validate_xray_input(pil_img)
    quality = validation["quality"]
    if not validation["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "not_xray_image",
                "message": "This does not appear to be a valid X-ray image.",
                "reasons": validation["reasons"],
                "quality": quality,
            },
        )

    if model is None:
        return JSONResponse(content=demo_predict(pil_img))

    original_rgb, batch = preprocess_image(pil_img, tta_passes=tta_passes)

    preds = model.predict(batch, verbose=0).reshape(-1)
    prob  = float(np.mean(preds))
    prediction_std = float(np.std(preds))
    decision = classify_probability(prob)
    label = decision["label"]
    gradcam_target = decision["gradcam_target"]
    conf  = prob if prob > 0.5 else 1.0 - prob

    try:
        heatmap     = make_gradcam_heatmap(batch[:1], target_class=gradcam_target)
        gradcam_b64 = to_b64(build_overlay(original_rgb, heatmap))
        heatmap_b64 = heatmap_to_rgba_b64(original_rgb, heatmap)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        gradcam_b64 = to_b64(original_rgb)
        heatmap_b64 = ""

    return JSONResponse(content={
        "label":         label,
        "label_type":    decision["label_type"],
        "risk_level":    decision["risk_level"],
        "recommendation": decision["recommendation"],
        "prob_fracture": round(prob, 4),
        "prob_normal":   round(1.0 - prob, 4),
        "confidence":    round(conf * 100, 1),
        "threshold":     THRESHOLD,
        "decision_thresholds": {
            "normal_max": UNCERTAIN_LOW,
            "review_range": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
            "fracture_min": UNCERTAIN_HIGH,
            "research_threshold": THRESHOLD,
        },
        "prediction_std": round(prediction_std, 4),
        "tta_passes":     int(tta_passes),
        "model_name":     MODEL_DISPLAY_NAME,
        "device_mode":    DEVICE_MODE,
        "is_uncertain":   decision["label_type"] == "uncertain",
        "quality":        quality,
        "gradcam_target": gradcam_target,
        "demo_mode":     False,
        "original_b64":  to_b64(original_rgb),
        "heatmap_b64":   heatmap_b64,
        "gradcam_b64":   gradcam_b64,
    })


@app.get("/health")
def health():
    return {
        "status":              "ok",
        "model_loaded":        model is not None,
        "backbone_feat_model": backbone_feat_model is not None,
        "conv_layer":          conv_layer_name,
        "threshold":           THRESHOLD,
        "uncertain_low":       UNCERTAIN_LOW,
        "uncertain_high":      UNCERTAIN_HIGH,
        "max_tta_passes":      MAX_TTA_PASSES,
        "default_tta_passes":  DEFAULT_TTA_PASSES,
        "device_mode":         DEVICE_MODE,
    }
