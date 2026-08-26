import os
import time
import io
import json
import random
from typing import Optional, List

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Load TFLite Engine ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "potato_quantized.tflite")
IMG_SIZE = 256
CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]

# Temperature scaling: calibrated on validation set via L-BFGS-B optimization
METADATA_PATH = os.path.join(os.path.dirname(__file__), "model", "class_names.json")
TEMPERATURE = 0.4662
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r") as f:
            _meta = json.load(f)
            TEMPERATURE = float(_meta.get("optimal_temperature", 0.4662))
            if "class_names" in _meta:
                CLASS_NAMES = _meta["class_names"]
    except Exception as e:
        print(f"Metadata load notice: {e}")

DISEASE_INFO = {
    "Early_Blight": {
        "emoji": "🟤",
        "name": "Early Blight",
        "pathogen": "Alternaria solani",
        "severity": "Moderate",
        "severity_level": 2,
        "color": "#f59e0b",
        "badge_class": "badge-warning",
        "description": "A destructive fungal disease characterized by dark brown concentric 'target board' spots on mature foliage, typically starting on lower leaves and moving upwards.",
        "symptoms": [
            "Concentric ringed brown/black target spots",
            "Yellow chlorotic halos surrounding leaf lesions",
            "Premature defoliation and leaf curling",
            "Dark, sunken stem lesions near base"
        ],
        "causes": [
            "Fungal pathogen Alternaria solani",
            "Warm temperatures (24°C – 29°C) with high humidity",
            "Prolonged leaf wetness from rain or overhead irrigation",
            "Plant stress and nitrogen deficiency"
        ],
        "treatment": [
            "Apply targeted fungicides (Chlorothalonil, Mancozeb, or Copper-based sprays)",
            "Prune and dispose of lower infected foliage promptly",
            "Avoid sprinkler/overhead irrigation; switch to drip watering",
            "Apply balanced fertilizer to boost plant vigor"
        ],
        "prevention": [
            "Plant certified disease-resistant potato seed varieties",
            "Practice 3-year crop rotation avoiding Solanaceae family",
            "Apply organic mulch to stop fungal soil splashback",
            "Ensure adequate row spacing for optimum canopy airflow"
        ],
        "urgent_alert": False
    },
    "Healthy": {
        "emoji": "🌿",
        "name": "Healthy Plant",
        "pathogen": "None (Optimum Foliage)",
        "severity": "None",
        "severity_level": 0,
        "color": "#10b981",
        "badge_class": "badge-success",
        "description": "The potato leaf displays pristine vitality. Foliage is crisp, uniformly pigmented, with vigorous vascular structure and no pathogenic lesions.",
        "symptoms": [
            "Uniform vibrant emerald green pigmentation",
            "Intact leaf margins and firm leaf cuticle",
            "No necrotic spots, water-soaking, or powdery mold",
            "Strong turgid petiole and upright leaf posture"
        ],
        "causes": [
            "Balanced soil nutrients (N-P-K & micronutrients)",
            "Optimal sunlight exposure (6-8 hours daily)",
            "Adequate soil drainage and regulated moisture",
            "Absence of foliar pathogens and pest vectors"
        ],
        "treatment": [
            "No chemical or corrective treatment required!",
            "Maintain consistent routine watering schedule",
            "Continue periodic scouting every 3–4 days",
            "Maintain balanced nutrient feeding during tuber bulking"
        ],
        "prevention": [
            "Keep soil well-drained and mulched",
            "Monitor regularly for aphid or beetle vectors",
            "Maintain companion planting (e.g. marigolds, basil)",
            "Conduct routine soil health checks"
        ],
        "urgent_alert": False
    },
    "Late_Blight": {
        "emoji": "🚨",
        "name": "Late Blight",
        "pathogen": "Phytophthora infestans",
        "severity": "CRITICAL ⚠️",
        "severity_level": 3,
        "color": "#ef4444",
        "badge_class": "badge-danger",
        "description": "A catastrophic water-mold (oomycete) infection capable of annihilating entire fields within 48–72 hours under humid conditions. Historical cause of the Great Famine.",
        "symptoms": [
            "Dark, rapidly enlarging water-soaked lesions",
            "White velvety fungal sporulation on leaf undersides",
            "Rapid necrosis turning foliage black and decaying",
            "Distinct pungent foul odor from rotting tissues"
        ],
        "causes": [
            "Oomycete Phytophthora infestans airborne sporangia",
            "Cool and damp conditions (10°C – 20°C)",
            "Relative humidity consistently above 90%",
            "Infected seed tubers or volunteer potato cull piles"
        ],
        "treatment": [
            "🚨 ACT IMMEDIATELY: Apply systemic fungicides (Metalaxyl, Cymoxanil, Mandipropamid)",
            "Severely cut and securely bag all infected vines (do NOT compost)",
            "Destroy cull piles and eliminate infected volunteer potatoes",
            "Notify local agricultural extension agent for regional blight tracking"
        ],
        "prevention": [
            "Always source certified disease-free seed tubers",
            "Apply preventative copper or chlorothalonil fungicides before rainy spells",
            "Eliminate all standing water and increase plant spacing",
            "Plant Late Blight resistant cultivars (e.g., Defender, Sarpo Mira)"
        ],
        "urgent_alert": True
    }
}

# ── Load Model ────────────────────────────────────────────────────────────────
def load_tflite_interpreter():
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=MODEL_PATH)
    except Exception:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        except Exception:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Preprocess PIL image for model inference.
    Uses LANCZOS resampling (higher quality than BILINEAR) for better
    detail preservation in real-world field photographs.
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _field_augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply a random field-realistic augmentation to an image.
    
    Covers the visual variations seen in real-world potato field photos:
    - Geometric: flips, rotations (handles any camera orientation)
    - Photometric: brightness, contrast, color saturation, sharpness
      (handles sun angle, cloud cover, camera settings)
    This diversity is critical because the PLD training dataset uses
    controlled studio lighting; real photos differ significantly.
    """
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
        lambda im: ImageEnhance.Brightness(im).enhance(rng.uniform(0.70, 1.35)),
        lambda im: ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.30)),
        lambda im: ImageEnhance.Color(im).enhance(rng.uniform(0.75, 1.30)),
        lambda im: ImageEnhance.Sharpness(im).enhance(rng.uniform(0.60, 1.60)),
    ]
    # Mix one geometric + one photometric for diverse coverage
    img = rng.choice(geometric_ops)(image)
    img = rng.choice(photometric_ops)(img)
    return img


def _temperature_scale(probs: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """Apply temperature scaling to sharpen softmax outputs.
    
    The model was trained with label_smoothing=0.1, which deliberately
    suppresses peak confidence to ~0.9 at training time. Temperature scaling
    (T < 1.0) reverses this at inference by sharpening the log-probability
    distribution before re-normalizing via softmax.
    
    References:
        Guo et al. (2017), 'On Calibration of Modern Neural Networks', ICML.
    """
    log_probs = np.log(np.maximum(probs, 1e-12))
    scaled = log_probs / temperature
    # Numerically stable softmax
    scaled -= np.max(scaled)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)


def run_inference(image: Image.Image, use_tta: bool = True, tta_passes: int = 9) -> tuple[np.ndarray, float]:
    """Run model inference with optional Test-Time Augmentation.
    
    TTA strategy:
    - Pass 0: original image (always included)
    - Passes 1..N: field-realistic augmentations (geometric + photometric)
    - Aggregation: geometric mean (superior to arithmetic for probabilities;
      down-weights outlier augmentations that confuse the model)
    - Post-processing: temperature scaling to recover confidence suppressed
      by label smoothing during training.
    """
    start_time = time.perf_counter()
    rng = random.Random()  # Independent RNG per call for thread safety
    
    if not use_tta or tta_passes <= 1:
        arr = preprocess_pil_image(image)
        interpreter.set_tensor(input_details[0]["index"], arr)
        interpreter.invoke()
        raw_probs = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float64)
        probs = _temperature_scale(raw_probs)
    else:
        # Build augmented image set: always start with original
        augmented_images = [image]
        while len(augmented_images) < tta_passes:
            augmented_images.append(_field_augment(image, rng))
        
        # Geometric mean accumulation (log-space for numerical stability)
        log_prob_accum = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        n_passes = len(augmented_images[:tta_passes])
        
        for aug_img in augmented_images[:tta_passes]:
            arr = preprocess_pil_image(aug_img)
            interpreter.set_tensor(input_details[0]["index"], arr)
            interpreter.invoke()
            out_prob = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float64)
            log_prob_accum += np.log(np.maximum(out_prob, 1e-12))
        
        # Geometric mean then re-normalize
        geo_mean = np.exp(log_prob_accum / n_passes)
        geo_mean = geo_mean / np.sum(geo_mean)
        
        # Apply temperature scaling after aggregation
        probs = _temperature_scale(geo_mean)
    
    # Final safety normalization
    probs = np.array(probs, dtype=np.float64)
    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum
        
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    return probs, duration_ms

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Potato Leaf Disease AI Classifier",
    description="Advanced Deep Learning Foliar Pathology Platform powered by EfficientNetB3 & TFLite",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "samples"), exist_ok=True)

# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "input_shape": input_details[0]["shape"].tolist(),
        "classes": CLASS_NAMES,
        "timestamp": time.time()
    }

@app.get("/api/info")
async def get_model_info():
    return {
        "model": {
            "name": "EfficientNetB3 Quantized",
            "architecture": "Convolutional Neural Network (EfficientNetB3)",
            "engine": "TensorFlow Lite (XNNPACK)",
            "input_resolution": f"{IMG_SIZE}x{IMG_SIZE}",
            "classes_count": len(CLASS_NAMES),
            "classes": CLASS_NAMES
        },
        "diseases": DISEASE_INFO
    }

@app.get("/api/samples")
async def list_sample_images():
    samples_dir = os.path.join(STATIC_DIR, "samples")
    sample_metadata = {
        "01_real_early_blight_1.jpg": {
            "name": "Early Blight (Real #1)",
            "expected_class": "Early_Blight",
            "emoji": "🟤",
            "tag": "Alternaria solani",
            "badge": "Early Blight"
        },
        "02_real_early_blight_2.jpg": {
            "name": "Early Blight (Real #2)",
            "expected_class": "Early_Blight",
            "emoji": "🟤",
            "tag": "Alternaria solani",
            "badge": "Early Blight"
        },
        "03_real_healthy_1.jpg": {
            "name": "Healthy Leaf (Real #1)",
            "expected_class": "Healthy",
            "emoji": "🌿",
            "tag": "Healthy Foliage",
            "badge": "Healthy"
        },
        "04_real_healthy_2.jpg": {
            "name": "Healthy Leaf (Real #2)",
            "expected_class": "Healthy",
            "emoji": "🌿",
            "tag": "Healthy Foliage",
            "badge": "Healthy"
        },
        "05_real_late_blight_1.jpg": {
            "name": "Late Blight (Real #1)",
            "expected_class": "Late_Blight",
            "emoji": "🚨",
            "tag": "Phytophthora infestans",
            "badge": "Late Blight"
        },
        "06_real_late_blight_2.jpg": {
            "name": "Late Blight (Real #2)",
            "expected_class": "Late_Blight",
            "emoji": "🚨",
            "tag": "Phytophthora infestans",
            "badge": "Late Blight"
        }
    }
    
    samples = []
    if os.path.exists(samples_dir):
        for f in sorted(os.listdir(samples_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                meta = sample_metadata.get(f, {
                    "name": f.rsplit(".", 1)[0].replace("_", " ").title(),
                    "expected_class": "Unknown",
                    "emoji": "🍃",
                    "tag": "PLD Dataset",
                    "badge": "Dataset Sample"
                })
                samples.append({
                    "id": f,
                    "name": meta["name"],
                    "expected_class": meta.get("expected_class"),
                    "emoji": meta.get("emoji", "🍃"),
                    "tag": meta.get("tag", "PLD"),
                    "badge": meta.get("badge", "Sample"),
                    "url": f"/static/samples/{f}"
                })
    return {
        "dataset_name": "Potato Disease Leaf Dataset (PLD)",
        "kaggle_url": "https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld",
        "samples": samples
    }

@app.post("/api/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    use_tta: bool = Form(True),
    tta_passes: int = Form(9),
    confidence_threshold: float = Form(70.0)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        original_size = {"width": image.width, "height": image.height}
        img_format = image.format or "JPEG"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
        
    probs, duration_ms = run_inference(image, use_tta=use_tta, tta_passes=tta_passes)
    
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence_pct = float(probs[pred_idx] * 100.0)
    
    probabilities_dict = {}
    for cls_name, prob_val in zip(CLASS_NAMES, probs):
        probabilities_dict[cls_name] = {
            "name": DISEASE_INFO[cls_name]["name"],
            "emoji": DISEASE_INFO[cls_name]["emoji"],
            "probability": round(float(prob_val * 100.0), 2),
            "raw_prob": float(prob_val),
            "color": DISEASE_INFO[cls_name]["color"]
        }
        
    disease_details = DISEASE_INFO[pred_class]
    is_low_confidence = confidence_pct < confidence_threshold
    
    return {
        "status": "success",
        "prediction": {
            "class_key": pred_class,
            "display_name": disease_details["name"],
            "pathogen": disease_details["pathogen"],
            "emoji": disease_details["emoji"],
            "confidence": round(confidence_pct, 2),
            "confidence_threshold": confidence_threshold,
            "is_low_confidence": is_low_confidence,
            "severity": disease_details["severity"],
            "severity_level": disease_details["severity_level"],
            "badge_class": disease_details["badge_class"],
            "color": disease_details["color"],
            "urgent_alert": disease_details["urgent_alert"] and confidence_pct >= 60.0
        },
        "probabilities": probabilities_dict,
        "diagnostics": {
            "description": disease_details["description"],
            "symptoms": disease_details["symptoms"],
            "causes": disease_details["causes"],
            "treatment": disease_details["treatment"],
            "prevention": disease_details["prevention"]
        },
        "meta": {
            "filename": file.filename,
            "image_size": original_size,
            "image_format": img_format,
            "inference_time_ms": round(duration_ms, 2),
            "tta_applied": use_tta,
            "tta_passes": tta_passes if use_tta else 1
        }
    }

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Potato Disease Detector Frontend Loading...</h1>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
