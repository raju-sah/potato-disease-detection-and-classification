import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import time
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🥔 Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "model/potato_quantized.tflite"
IMG_SIZE    = 256
CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]

DISEASE_INFO = {
    "Early_Blight": {
        "emoji"      : "🟤",
        "full_name"  : "Early Blight (Alternaria solani)",
        "severity"   : "Moderate",
        "severity_color": "orange",
        "description": "A fungal disease causing dark brown spots with concentric rings (target-like pattern) on leaves. It starts on older/lower leaves and moves upward.",
        "symptoms"   : ["Dark brown circular spots with yellow halos", "Concentric ring pattern inside spots", "Leaves turn yellow and drop early", "Stem lesions near soil line"],
        "causes"     : ["Fungus: Alternaria solani", "Warm & humid weather (24–29°C)", "Overcrowded plants", "Poor air circulation"],
        "treatment"  : ["Apply chlorothalonil or mancozeb fungicide", "Remove and destroy infected leaves", "Avoid overhead irrigation", "Rotate crops every season"],
        "prevention" : ["Use certified disease-free seeds", "Maintain plant spacing", "Apply mulch to prevent soil splash", "Water at the base of plants"],
        "color"      : "#e67e22",
        "bg_color"   : "#fef9f0"
    },
    "Healthy": {
        "emoji"      : "🟢",
        "full_name"  : "Healthy Potato Plant",
        "severity"   : "None",
        "severity_color": "green",
        "description": "The plant shows no signs of disease. Leaves are vibrant green, firm, and fully intact with no spots, lesions, or discolouration.",
        "symptoms"   : ["Vibrant green leaves", "No spots or lesions", "Strong upright stems", "Uniform leaf texture"],
        "causes"     : ["Optimal growing conditions", "Good soil nutrition", "Adequate watering", "Proper sunlight"],
        "treatment"  : ["No treatment needed!", "Continue routine care", "Monitor regularly for early signs", "Maintain fertilisation schedule"],
        "prevention" : ["Continue current practices", "Regular crop inspection", "Balanced NPK fertilisation", "Proper irrigation management"],
        "color"      : "#27ae60",
        "bg_color"   : "#f0fef4"
    },
    "Late_Blight": {
        "emoji"      : "⚫",
        "full_name"  : "Late Blight (Phytophthora infestans)",
        "severity"   : "SEVERE ⚠️",
        "severity_color": "red",
        "description": "A highly destructive oomycete disease. It caused the Irish Potato Famine (1840s). Can destroy an entire crop within days under favourable conditions.",
        "symptoms"   : ["Dark water-soaked lesions on leaves", "White fuzzy growth on leaf undersides", "Brown/black patches spreading rapidly", "Foul smell from infected tissue"],
        "causes"     : ["Pathogen: Phytophthora infestans", "Cool & wet weather (10–20°C)", "High humidity (>90%)", "Infected seed tubers"],
        "treatment"  : ["⚠️ Act IMMEDIATELY", "Apply metalaxyl or cymoxanil fungicide", "Remove and bag all infected plants", "Do NOT compost infected material", "Notify local agricultural authority"],
        "prevention" : ["Use resistant varieties", "Plant certified disease-free tubers", "Avoid wet foliage at night", "Apply preventive copper-based fungicides"],
        "color"      : "#e74c3c",
        "bg_color"   : "#fef0f0"
    }
}

# ── Load TFLite model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# ── Preprocess image ──────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── TFLite inference ──────────────────────────────────────────────────────────
def predict(interpreter, image: Image.Image):
    inp  = interpreter.get_input_details()
    out  = interpreter.get_output_details()
    arr  = preprocess(image)
    interpreter.set_tensor(inp[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(out[0]["index"])[0]
    return probs

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #27ae60, #2ecc71, #f39c12);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .subtitle {
        text-align: center; color: #7f8c8d;
        font-size: 1.1rem; margin-bottom: 30px;
    }
    .result-card {
        border-radius: 16px; padding: 24px;
        border-left: 6px solid; margin: 16px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-box {
        background: #f8f9fa; border-radius: 12px;
        padding: 16px; text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
    .section-header {
        font-size: 1.2rem; font-weight: 700;
        margin: 20px 0 10px; padding-bottom: 6px;
        border-bottom: 2px solid #e9ecef;
    }
    .tag {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; margin: 3px;
    }
    .stButton > button {
        width: 100%; border-radius: 10px;
        font-weight: 700; font-size: 1rem;
        padding: 12px; transition: all 0.3s;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🥔 About This App")
    st.info(
        "This app uses **EfficientNetB3** (converted to TFLite) "
        "to classify potato leaf diseases in real-time."
    )

    st.markdown("### 🌿 Detectable Classes")
    for cls, info in DISEASE_INFO.items():
        st.markdown(
            f"<span style='color:{info['color']};font-weight:700'>"
            f"{info['emoji']} {cls.replace('_',' ')}</span> — {info['severity']}",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_tta    = st.toggle("🔀 Enable TTA (Test Time Augmentation)", value=False)
    tta_passes  = st.slider("TTA Passes", 3, 15, 5, disabled=not show_tta)
    conf_thresh = st.slider("⚠️ Low Confidence Threshold (%)", 50, 90, 70)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Architecture | EfficientNetB3 |
    | Input Size | 256×256 |
    | Format | TFLite (Quantized) |
    | Classes | 3 |
    | Dataset | PLD Dataset |
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & TensorFlow Lite")


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🥔 Potato Leaf Disease Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a potato leaf image to detect disease using AI — powered by EfficientNetB3</p>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────
try:
    interpreter = load_model()
    st.success("✅ Model loaded successfully!", icon="🤖")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ── Upload section ────────────────────────────
st.markdown("---")
col_upload, col_info = st.columns([1.2, 1])

with col_upload:
    st.markdown("### 📤 Upload Leaf Image")
    uploaded = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit photo of a single potato leaf"
    )

with col_info:
    st.markdown("### 📌 Tips for Best Results")
    st.markdown("""
    - 📷 Use a **clear, well-lit** photo
    - 🍃 Make sure **one leaf** fills most of the frame
    - 🚫 Avoid blurry or dark images
    - 📐 Any aspect ratio works — app auto-resizes
    - 🌿 Works best with **natural lighting**
    """)

# ── Prediction ────────────────────────────────
if uploaded is not None:
    image = Image.open(uploaded)

    st.markdown("---")
    col_img, col_result = st.columns([1, 1.4])

    with col_img:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_column_width=True, caption=f"📁 {uploaded.name}")
        st.markdown(f"**Size:** {image.size[0]}×{image.size[1]} px  |  **Format:** {image.format or 'N/A'}")

    with col_result:
        st.markdown("### 🔍 Analysis")

        with st.spinner("🧠 Analysing leaf... please wait"):
            time.sleep(0.3)

            if show_tta:
                # Simple TTA: flip + rotate variations
                probs_sum = predict(interpreter, image)
                for _ in range(tta_passes - 1):
                    import random
                    aug_img = image.transpose(
                        random.choice([Image.FLIP_LEFT_RIGHT,
                                       Image.FLIP_TOP_BOTTOM,
                                       Image.ROTATE_90,
                                       Image.ROTATE_180])
                    )
                    probs_sum = probs_sum + predict(interpreter, aug_img)
                probs = probs_sum / tta_passes
            else:
                probs = predict(interpreter, image)

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        info       = DISEASE_INFO[pred_class]

        # ── Result card ───────────────────────
        st.markdown(
            f"""<div class="result-card" style="
                background:{info['bg_color']};
                border-color:{info['color']};">
                <h2 style="color:{info['color']};margin:0">
                    {info['emoji']} {pred_class.replace('_',' ')}
                </h2>
                <p style="color:#555;margin:6px 0 0">{info['full_name']}</p>
            </div>""",
            unsafe_allow_html=True
        )

        # ── Metrics row ───────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""<div class="metric-box">
                <div class="metric-value" style="color:{info['color']}">{confidence:.1f}%</div>
                <div class="metric-label">Confidence</div>
                </div>""", unsafe_allow_html=True
            )
        with m2:
            sev_colors = {"None":"#27ae60","Moderate":"#e67e22","SEVERE ⚠️":"#e74c3c"}
            sev_color  = sev_colors.get(info["severity"], "#333")
            st.markdown(
                f"""<div class="metric-box">
                <div class="metric-value" style="color:{sev_color};font-size:1.3rem">
                    {info['severity']}</div>
                <div class="metric-label">Severity</div>
                </div>""", unsafe_allow_html=True
            )
        with m3:
            mode = f"TTA ×{tta_passes}" if show_tta else "Standard"
            st.markdown(
                f"""<div class="metric-box">
                <div class="metric-value" style="font-size:1.1rem">{mode}</div>
                <div class="metric-label">Inference Mode</div>
                </div>""", unsafe_allow_html=True
            )

        # ── Low confidence warning ─────────────
        if confidence < conf_thresh:
            st.warning(
                f"⚠️ Confidence is below {conf_thresh}%. "
                "Consider uploading a clearer image for more reliable results."
            )

    # ── Probability chart ─────────────────────
    st.markdown("---")
    st.markdown("### 📊 Class Probability Distribution")

    chart_col, detail_col = st.columns([1.2, 1])

    with chart_col:
        colors_chart = [DISEASE_INFO[c]["color"] for c in CLASS_NAMES]
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in probs],
            y=[c.replace("_", " ") for c in CLASS_NAMES],
            orientation="h",
            marker=dict(color=colors_chart, line=dict(color="white", width=1.5)),
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>"
        ))
        fig.update_layout(
            xaxis=dict(title="Probability (%)", range=[0, 115]),
            yaxis=dict(title=""),
            height=280,
            margin=dict(l=10, r=60, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13)
        )
        fig.add_vline(x=50, line_dash="dot", line_color="gray", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with detail_col:
        st.markdown("**All Class Probabilities:**")
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
            c_info = DISEASE_INFO[cls]
            pct    = prob * 100
            st.markdown(f"{c_info['emoji']} **{cls.replace('_',' ')}**")
            st.progress(float(prob), text=f"{pct:.2f}%")

    # ── Disease detail tabs ───────────────────
    st.markdown("---")
    st.markdown(f"### 🌿 About: {pred_class.replace('_', ' ')}")
    st.markdown(f"*{info['description']}*")

    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Symptoms", "🦠 Causes", "💊 Treatment", "🛡️ Prevention"])

    with tab1:
        for s in info["symptoms"]:
            st.markdown(f"- {s}")
    with tab2:
        for c in info["causes"]:
            st.markdown(f"- {c}")
    with tab3:
        for t in info["treatment"]:
            st.markdown(f"- {t}")
    with tab4:
        for p in info["prevention"]:
            st.markdown(f"- {p}")

    # ── Late Blight emergency alert ───────────
    if pred_class == "Late_Blight" and confidence > 70:
        st.error(
            "🚨 **EMERGENCY ALERT — Late Blight Detected!**\n\n"
            "This disease can destroy your **entire crop within days**. "
            "Immediately isolate affected plants and consult your local agricultural officer.",
            icon="🚨"
        )

else:
    # ── Empty state ───────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    for col, (cls, info) in zip([c1, c2, c3], DISEASE_INFO.items()):
        with col:
            st.markdown(
                f"""<div style="background:{info['bg_color']};border-radius:14px;
                padding:20px;border-left:5px solid {info['color']};
                text-align:center;">
                <div style="font-size:2.5rem">{info['emoji']}</div>
                <div style="font-weight:800;font-size:1.1rem;color:{info['color']}">
                    {cls.replace('_',' ')}</div>
                <div style="font-size:0.85rem;color:#666;margin-top:8px">
                    Severity: <b>{info['severity']}</b></div>
                </div>""",
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Upload a potato leaf image above to get started!", icon="🍃")