import os
import urllib.request
import urllib.parse
from PIL import Image
from server import run_inference, CLASS_NAMES

REPO_BASE = "https://raw.githubusercontent.com/sanils2002/Potato-Leaf-Disease/main/training/PlantVillage"

REAL_SAMPLES = [
    {
        "filename": "01_real_early_blight_1.jpg",
        "label": "Real Early Blight #1",
        "expected": "Early_Blight",
        "remote_path": "Potato___Early_blight/001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG"
    },
    {
        "filename": "02_real_early_blight_2.jpg",
        "label": "Real Early Blight #2",
        "expected": "Early_Blight",
        "remote_path": "Potato___Early_blight/002a55fb-7a3d-4a3a-aca8-ce2d5ebc6925___RS_Early.B 8170.JPG"
    },
    {
        "filename": "03_real_healthy_1.jpg",
        "label": "Real Healthy #1",
        "expected": "Healthy",
        "remote_path": "Potato___healthy/00fc8eac-fbef-4c7f-8f88-9b1deaa8a0b8___RS_HL 4154.JPG"
    },
    {
        "filename": "04_real_healthy_2.jpg",
        "label": "Real Healthy #2",
        "expected": "Healthy",
        "remote_path": "Potato___healthy/03b53c7a-9774-4b53-a74e-5a0225d304f5___RS_HL 1777.JPG"
    },
    {
        "filename": "05_real_late_blight_1.jpg",
        "label": "Real Late Blight #1",
        "expected": "Late_Blight",
        "remote_path": "Potato___Late_blight/0051e5e8-d1c4-4a84-b0f6-6386b684c792___RS_LB 4640.JPG"
    },
    {
        "filename": "06_real_late_blight_2.jpg",
        "label": "Real Late Blight #2",
        "expected": "Late_Blight",
        "remote_path": "Potato___Late_blight/006955e3-473d-49b6-b029-ed22c1e457ab___RS_LB 4513.JPG"
    }
]

os.makedirs("static/samples", exist_ok=True)

# Clean out old synthetic images if present
for old_f in ["01_healthy_leaf.jpg", "02_early_blight.jpg", "03_late_blight.jpg"]:
    old_p = os.path.join("static/samples", old_f)
    if os.path.exists(old_p):
        os.remove(old_p)

print("Downloading real PLD / PlantVillage dataset images...")
for item in REAL_SAMPLES:
    encoded_path = urllib.parse.quote(item["remote_path"])
    full_url = f"{REPO_BASE}/{encoded_path}"
    dest_path = os.path.join("static/samples", item["filename"])
    
    req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
            out_f.write(resp.read())
        
        # Test inference
        img = Image.open(dest_path)
        probs, t = run_inference(img)
        pred = CLASS_NAMES[probs.argmax()]
        conf = probs.max() * 100.0
        print(f"✅ {item['filename']} -> Predicted: {pred} ({conf:.2f}%) in {t:.1f}ms [Expected: {item['expected']}]")
    except Exception as e:
        print(f"❌ Error for {item['filename']} ({full_url}): {e}")
