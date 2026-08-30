import os
import time
import random
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from backend.config import CLASS_NAMES, TEMPERATURE, IMG_SIZE, MODEL_ZOO_DIR, DEFAULT_MODEL_PATH

def create_interpreter(path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=path)
    except Exception:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=path)
        except Exception:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=path)
            
    interpreter.allocate_tensors()
    return interpreter

INTERPRETERS: Dict[str, Any] = {}

# Load all models in zoo
for key in ["densenet121", "convnext_tiny", "efficientnet_v2s", "resnet50", "mobilenet_v3"]:
    model_file = os.path.join(MODEL_ZOO_DIR, f"{key}.tflite")
    if os.path.exists(model_file):
        try:
            INTERPRETERS[key] = create_interpreter(model_file)
            print(f"✅ Loaded model zoo interpreter: {key}")
        except Exception as e:
            print(f"❌ Failed to load {key}: {e}")

# Fallback default interpreter
if "densenet121" not in INTERPRETERS and os.path.exists(DEFAULT_MODEL_PATH):
    try:
        INTERPRETERS["densenet121"] = create_interpreter(DEFAULT_MODEL_PATH)
    except:
        pass

def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Preprocess PIL image for model inference with LANCZOS resampling."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _field_augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply a random field-realistic augmentation to an image."""
    geometric_ops = [
        lambda im: ImageOps.mirror(im),
        lambda im: ImageOps.flip(im),
        lambda im: im.rotate(90),
        lambda im: im.rotate(180),
        lambda im: im.rotate(270),
        lambda im: ImageOps.mirror(im.rotate(90)),
        lambda im: ImageOps.mirror(im.rotate(270)),
    ]
    photometric_ops = [
        lambda im: ImageEnhance.Brightness(im).enhance(rng.uniform(0.80, 1.25)),
        lambda im: ImageEnhance.Contrast(im).enhance(rng.uniform(0.80, 1.25)),
        lambda im: ImageEnhance.Color(im).enhance(rng.uniform(0.80, 1.25)),
        lambda im: ImageEnhance.Sharpness(im).enhance(rng.uniform(0.70, 1.40)),
    ]
    img = rng.choice(geometric_ops)(image)
    img = rng.choice(photometric_ops)(img)
    return img

def _temperature_scale(probs: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """Apply temperature scaling to calibrate softmax outputs."""
    log_probs = np.log(np.maximum(probs, 1e-12))
    scaled = log_probs / temperature
    scaled -= np.max(scaled)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)

def _predict_single_interpreter(interp: Any, arr: np.ndarray) -> np.ndarray:
    """Run forward pass on a single TFLite interpreter."""
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]
    interp.set_tensor(inp, arr)
    interp.invoke()
    return interp.get_tensor(out)[0].astype(np.float64)

def run_inference(
    image: Image.Image, 
    model_id: str = "ensemble", 
    use_tta: bool = True, 
    tta_passes: int = 9
) -> tuple[np.ndarray, float, Dict[str, Any]]:
    """Run model inference with optional TTA and Multi-Model Ensemble support."""
    start_time = time.perf_counter()
    rng = random.Random()
    ensemble_breakdown: Dict[str, Any] = {}

    def get_augmented_arrays() -> List[np.ndarray]:
        if not use_tta or tta_passes <= 1:
            return [preprocess_pil_image(image)]
        aug_imgs = [image]
        while len(aug_imgs) < tta_passes:
            aug_imgs.append(_field_augment(image, rng))
        return [preprocess_pil_image(img) for img in aug_imgs[:tta_passes]]

    arrays = get_augmented_arrays()

    def eval_raw_probs(interp: Any) -> np.ndarray:
        if len(arrays) == 1:
            return _predict_single_interpreter(interp, arrays[0])
        log_acc = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for arr in arrays:
            raw = _predict_single_interpreter(interp, arr)
            log_acc += np.log(np.maximum(raw, 1e-12))
        geo = np.exp(log_acc / len(arrays))
        return geo / np.sum(geo)

    if model_id == "ensemble":
        dense_interp = INTERPRETERS.get("densenet121")
        conv_interp = INTERPRETERS.get("convnext_tiny")
        eff_interp = INTERPRETERS.get("efficientnet_v2s")

        r_dense = eval_raw_probs(dense_interp) if dense_interp else np.array([0.33, 0.34, 0.33])
        r_conv = eval_raw_probs(conv_interp) if conv_interp else r_dense
        r_eff = eval_raw_probs(eff_interp) if eff_interp else r_dense

        # Soft-voting on raw output distributions
        raw_ens = 0.45 * r_dense + 0.35 * r_conv + 0.20 * r_eff
        raw_ens = raw_ens / np.sum(raw_ens)
        probs = _temperature_scale(raw_ens, temperature=0.55)

        # Calibrated member breakdowns for UI inspection
        cal_dense = _temperature_scale(r_dense, temperature=0.60)
        cal_conv = _temperature_scale(r_conv, temperature=0.60)
        cal_eff = _temperature_scale(r_eff, temperature=0.60)

        ensemble_breakdown = {
            "densenet121": {
                "name": "DenseNet-121",
                "pred_class": CLASS_NAMES[np.argmax(cal_dense)],
                "confidence": round(float(np.max(cal_dense) * 100.0), 1),
                "weight": "45%"
            },
            "convnext_tiny": {
                "name": "ConvNeXt-Tiny",
                "pred_class": CLASS_NAMES[np.argmax(cal_conv)],
                "confidence": round(float(np.max(cal_conv) * 100.0), 1),
                "weight": "35%"
            },
            "efficientnet_v2s": {
                "name": "EfficientNetV2-S",
                "pred_class": CLASS_NAMES[np.argmax(cal_eff)],
                "confidence": round(float(np.max(cal_eff) * 100.0), 1),
                "weight": "20%"
            }
        }
    else:
        chosen_interp = INTERPRETERS.get(model_id) or INTERPRETERS.get("densenet121")
        if not chosen_interp:
            raise ValueError(f"Model {model_id} not available in model zoo.")
        raw_single = eval_raw_probs(chosen_interp)
        probs = _temperature_scale(raw_single, temperature=TEMPERATURE)

    probs = np.array(probs, dtype=np.float64)
    probs = probs / np.sum(probs)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    return probs, duration_ms, ensemble_breakdown
