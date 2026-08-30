import os
import io
import json

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import BASE_DIR, STATIC_DIR, CLASS_NAMES, TEMPERATURE, DISEASE_INFO, AVAILABLE_MODELS
from backend.inference import run_inference
from backend.xai import generate_gradcam_payload

# Update models with dynamic benchmark metadata if available
METADATA_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
            benchmark_results = meta.get("benchmark_results", [])
            for res in benchmark_results:
                arch = res.get("Architecture")
                if arch == "DenseNet121" and "densenet121" in AVAILABLE_MODELS:
                    AVAILABLE_MODELS["densenet121"]["test_acc"] = f"{res.get('ID Accuracy (%)')}%"
                    AVAILABLE_MODELS["densenet121"]["test_f1"] = f"{res.get('ID Macro-F1 (%)')}%"
                elif arch == "ConvNeXtTiny" and "convnext_tiny" in AVAILABLE_MODELS:
                    AVAILABLE_MODELS["convnext_tiny"]["test_acc"] = f"{res.get('ID Accuracy (%)')}%"
                    AVAILABLE_MODELS["convnext_tiny"]["test_f1"] = f"{res.get('ID Macro-F1 (%)')}%"
                elif arch == "EfficientNetV2S" and "efficientnet_v2s" in AVAILABLE_MODELS:
                    AVAILABLE_MODELS["efficientnet_v2s"]["test_acc"] = f"{res.get('ID Accuracy (%)')}%"
                    AVAILABLE_MODELS["efficientnet_v2s"]["test_f1"] = f"{res.get('ID Macro-F1 (%)')}%"
                elif arch == "ResNet50" and "resnet50" in AVAILABLE_MODELS:
                    AVAILABLE_MODELS["resnet50"]["test_acc"] = f"{res.get('ID Accuracy (%)')}%"
                    AVAILABLE_MODELS["resnet50"]["test_f1"] = f"{res.get('ID Macro-F1 (%)')}%"
                elif arch == "MobileNetV3" and "mobilenet_v3" in AVAILABLE_MODELS:
                    AVAILABLE_MODELS["mobilenet_v3"]["test_acc"] = f"{res.get('ID Accuracy (%)')}%"
                    AVAILABLE_MODELS["mobilenet_v3"]["test_f1"] = f"{res.get('ID Macro-F1 (%)')}%"
    except Exception as e:
        print(f"Error loading benchmark metrics: {e}")

app = FastAPI(
    title="Potato Leaf Disease AI Classifier & Research Benchmark",
    description="Multi-Architecture Deep Learning Foliar Pathology Platform & Experimental Suite",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_engine": "Multi-Model TFLite Zoo (Float16 Quantized)",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "active_default": "ensemble",
        "classes": CLASS_NAMES,
        "calibrated_temperature": TEMPERATURE,
    }

@app.get("/api/models")
async def get_models():
    """Returns the list of available model architectures and their benchmark metadata."""
    return {
        "active_default": "ensemble",
        "models": list(AVAILABLE_MODELS.values())
    }

@app.get("/api/disease-info")
async def get_disease_info():
    return DISEASE_INFO

@app.get("/api/samples")
async def get_preset_samples():
    samples_dir = os.path.join(STATIC_DIR, "samples")
    samples = []
    sample_meta = {
        "01_real_early_blight_1.jpg": {
            "name": "Early Blight (Real #1)",
            "expected_class": "Early_Blight",
            "emoji": "🟤",
            "badge": "Early Blight",
            "tag": "Alternaria solani",
            "desc": "Concentric dark brown target spot lesions on lower canopy leaf"
        },
        "02_real_early_blight_2.jpg": {
            "name": "Early Blight (Real #2)",
            "expected_class": "Early_Blight",
            "emoji": "🟤",
            "badge": "Early Blight",
            "tag": "Alternaria solani",
            "desc": "Spreading chlorotic halo surrounding necrotic concentric rings"
        },
        "03_real_healthy_1.jpg": {
            "name": "Healthy Leaf (Real #1)",
            "expected_class": "Healthy",
            "emoji": "🌿",
            "badge": "Healthy",
            "tag": "Vigorous Foliage",
            "desc": "Pristine leaf cuticle with vibrant green chlorophyll pigmentation"
        },
        "04_real_healthy_2.jpg": {
            "name": "Healthy Leaf (Real #2)",
            "expected_class": "Healthy",
            "emoji": "🌿",
            "badge": "Healthy",
            "tag": "Vigorous Foliage",
            "desc": "Intact foliar vascular network without any pathogen spots"
        },
        "05_real_late_blight_1.jpg": {
            "name": "Late Blight (Real #1)",
            "expected_class": "Late_Blight",
            "emoji": "🚨",
            "badge": "Late Blight",
            "tag": "Phytophthora infestans",
            "desc": "Large water-soaked necrotized lesion rapidly consuming leaf margin"
        },
        "06_real_late_blight_2.jpg": {
            "name": "Late Blight (Real #2)",
            "expected_class": "Late_Blight",
            "emoji": "🚨",
            "badge": "Late Blight",
            "tag": "Phytophthora infestans",
            "desc": "Systemic dark blighted necrosis with tissue decay"
        }
    }
    
    if os.path.exists(samples_dir):
        files = sorted(os.listdir(samples_dir))
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                meta = sample_meta.get(f, {
                    "name": f.replace("_", " ").replace(".jpg", "").capitalize(),
                    "expected_class": "Unknown",
                    "emoji": "🍃",
                    "badge": "Sample",
                    "tag": "PLD",
                    "desc": "Potato leaf sample"
                })
                samples.append({
                    "id": f,
                    "name": meta["name"],
                    "expected_class": meta.get("expected_class"),
                    "emoji": meta.get("emoji", "🍃"),
                    "tag": meta.get("tag", "PLD"),
                    "badge": meta.get("badge", "Sample"),
                    "desc": meta.get("desc", ""),
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
    model_id: str = Form("ensemble"),
    use_tta: bool = Form(True),
    tta_passes: int = Form(9),
    confidence_threshold: float = Form(70.0)
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        original_size = {"width": image.width, "height": image.height}
        img_format = image.format or "JPEG"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
        
    probs, duration_ms, ensemble_breakdown = run_inference(
        image, 
        model_id=model_id, 
        use_tta=use_tta, 
        tta_passes=tta_passes
    )
    
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
    active_model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS["ensemble"])
    
    # Compute Grad-CAM Explainability Saliency
    explainability = generate_gradcam_payload(image, model_id=model_id, target_class_idx=pred_idx)

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
        "explainability": explainability,
        "model": {
            "id": active_model_info["id"],
            "name": active_model_info["name"],
            "paradigm": active_model_info["paradigm"],
            "params_m": active_model_info["params_m"],
            "test_acc": active_model_info["test_acc"],
            "badge": active_model_info["badge"],
            "emoji": active_model_info["emoji"],
            "ensemble_breakdown": ensemble_breakdown
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
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
