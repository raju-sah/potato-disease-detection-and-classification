import os
import json
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_ZOO_DIR = os.path.join(BASE_DIR, "model", "zoo")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "potato_quantized.tflite")
IMG_SIZE = 256
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Defaults
CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]
TEMPERATURE = 0.4662
METADATA_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

def load_metadata():
    global CLASS_NAMES, TEMPERATURE
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                _meta = json.load(f)
                TEMPERATURE = float(_meta.get("optimal_temperature", 0.4662))
                if "class_names" in _meta:
                    CLASS_NAMES = _meta["class_names"]
        except Exception as e:
            print(f"Metadata load notice: {e}")

load_metadata()

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
            "Sanitize pruning tools between field blocks"
        ],
        "urgent_alert": False
    },
    "Late_Blight": {
        "emoji": "🚨",
        "name": "Late Blight",
        "pathogen": "Phytophthora infestans",
        "severity": "Severe / Critical",
        "severity_level": 3,
        "color": "#ef4444",
        "badge_class": "badge-danger",
        "description": "A devastating oomycete pathogen capable of destroying entire potato fields within 7–10 days under cool, humid conditions. Responsible for the historic Great Irish Famine.",
        "symptoms": [
            "Water-soaked dark lesions spreading rapidly on foliage",
            "White fungal sporulation/fuzz on leaf undersides in high humidity",
            "Rapid systemic tissue necrosis and foliar collapse",
            "Foul odor in canopy from decomposing necrotic tissue"
        ],
        "causes": [
            "Oomycete pathogen Phytophthora infestans",
            "Cool to moderate temperatures (10°C – 20°C) with relative humidity > 90%",
            "Infected seed tubers or volunteer potato cull piles",
            "Airborne sporangia carried on wind currents"
        ],
        "treatment": [
            "IMMEDIATE ACTION REQUIRED: Apply systemic curative fungicides (Metalaxyl, Cymoxanil, Dimethomorph)",
            "Remove, bag, and bury/destroy heavily infected plants — DO NOT COMPOST",
            "Establish a mandatory 5-day protective spray schedule for surrounding rows",
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

AVAILABLE_MODELS = {
    "ensemble": {
        "id": "ensemble",
        "name": "Multi-Model Soft-Voting Ensemble",
        "paradigm": "Tri-Model Soft Voting (DenseNet + ConvNeXt + EfficientNet)",
        "badge": "Highest Reliability",
        "emoji": "🔮",
        "params_m": 56.00,
        "test_acc": "99.8%",
        "test_f1": "99.8%",
        "recommended": True,
        "description": "Weighted consensus (45% DenseNet-121 + 35% ConvNeXt-Tiny + 20% EfficientNetV2-S). Resolves individual architecture blind spots and achieves 100% empirical sample accuracy."
    },
    "densenet121": {
        "id": "densenet121",
        "name": "DenseNet-121 (Dense Feature Reuse)",
        "paradigm": "Dense Connectivity / Feature Concatenation",
        "badge": "Top Single Model",
        "emoji": "🏆",
        "params_m": 7.31,
        "test_acc": "99.75%",
        "test_f1": "99.73%",
        "recommended": False,
        "description": "Connects all layers directly, preserving fine-grained fungal texture cues and subtle chlorotic halos with high parameter efficiency."
    },
    "convnext_tiny": {
        "id": "convnext_tiny",
        "name": "ConvNeXt-Tiny (Modernized ConvNet)",
        "paradigm": "7x7 Depthwise Separable Convolutions",
        "badge": "Macro-Lesion Specialist",
        "emoji": "💎",
        "params_m": 28.02,
        "test_acc": "64.69%",
        "test_f1": "65.22%",
        "recommended": False,
        "description": "Modern inverted bottleneck design with large 7x7 receptive fields, superior at capturing wide concentric ring target boards."
    },
    "efficientnet_v2s": {
        "id": "efficientnet_v2s",
        "name": "EfficientNetV2-S (Compound Scaled)",
        "paradigm": "Fused-MBConv & Progressive Neural Architecture Search",
        "badge": "Fast Compound Scaling",
        "emoji": "⚡",
        "params_m": 20.67,
        "test_acc": "64.44%",
        "test_f1": "62.29%",
        "recommended": False,
        "description": "Combines Fused-MBConv and regular MBConv layers with progressive learning for balanced foliar feature representations."
    },
    "resnet50": {
        "id": "resnet50",
        "name": "ResNet-50 (Residual Baseline)",
        "paradigm": "Residual Skip Connections (Addition)",
        "badge": "Classical Baseline",
        "emoji": "🏛️",
        "params_m": 24.12,
        "test_acc": "44.20%",
        "test_f1": "29.10%",
        "recommended": False,
        "description": "Classical deep residual network providing identity shortcut mapping across 50 convolution layers."
    },
    "mobilenet_v3": {
        "id": "mobilenet_v3",
        "name": "MobileNetV3-Large (Lightweight Edge)",
        "paradigm": "Hardware-Aware NAS & Squeeze-and-Excitation",
        "badge": "Ultra-Fast Edge Mobile",
        "emoji": "📱",
        "params_m": 3.25,
        "test_acc": "40.74%",
        "test_f1": "21.05%",
        "recommended": False,
        "description": "Optimized for mobile CPU/edge hardware with Hard-Swish activations and squeeze-and-excitation blocks."
    }
}
