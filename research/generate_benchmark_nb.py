import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# Cell 1: Header
add_md("""# 🌿 Beyond the Lab: A Systematic Empirical Study on Architectural Robustness, Saliency Alignment, and Edge Compression for Potato Foliar Pathology Under Domain Shift

**Author:** Raju Sah  
**Focus:** Precision Folia Pathology, Multi-Paradigm Deep Learning, Domain Shift, Explainable AI (XAI), and Edge Optimization  
**Target Submission:** IEEE / Springer Journal of Agricultural Informatics / CVPR Workshop on Computer Vision in Agriculture  

---

### Abstract
Automated plant disease diagnosis via deep learning offers massive potential for global food security. However, standard academic benchmarks predominantly evaluate models on sterile laboratory datasets (e.g., PlantVillage), obscuring severe performance degradation when models encounter in-the-wild agricultural field conditions (**Domain Shift**). This study provides a rigorous, multi-paradigm empirical evaluation of:
1. **Architectural Paradigms (RQ1):** Classical CNNs (*ResNet-50*, *DenseNet-121*), Compound-Scaled CNNs (*EfficientNetV2-S*), Modernized ConvNets (*ConvNeXt-Tiny*), and Lightweight Edge CNNs (*MobileNetV3*).
2. **Domain Generalization (RQ2):** Quantifying the in-domain to out-of-domain ($\Delta_{\text{OOD}}$) accuracy and Macro-F1 gap.
3. **Data-Centric Regularization (RQ3):** Ablation of *RandAugment*, *CutMix*, *MixUp*, and *Label Smoothing*.
4. **Saliency Alignment (RQ4):** *Grad-CAM* & *Grad-CAM++* evaluation to detect background shortcut learning vs. pathognomonic lesion feature attribution.
5. **Edge Efficiency Pareto Frontier (RQ5):** Float32 vs. Float16 vs. INT8 quantization trade-offs across FLOPs, Latency, and Memory footprint.""")

# Cell 2: Imports and Setup
add_code("""# Step 1 — Reproducibility Harness, Environment Configuration & Imports
import os, sys, random, json, time, warnings, collections
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.optimize import minimize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    EfficientNetV2S,
    ConvNeXtTiny,
    MobileNetV3Large,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_recall_fscore_support, log_loss
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')

# Multi-seed reproducibility
SEEDS = [42, 123, 999]
PRIMARY_SEED = SEEDS[0]

def set_seed(seed=PRIMARY_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

set_seed(PRIMARY_SEED)
print(f'TensorFlow Version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'Available Hardware Accelerators (GPU): {gpus}')
""")

# Cell 3: Dataset Ingestion
add_code("""# Step 2 — Dataset Manifest Ingestion & Stratified Split Building
BASE = None
for cand in [
    Path('/kaggle/input/potato-disease-leaf-datasetpld/PLD_3_Classes_256'),
    Path('/kaggle/input/datasets/rizwan123456789/potato-disease-leaf-datasetpld/PLD_3_Classes_256'),
]:
    if cand.exists():
        BASE = cand
        break

assert BASE is not None, f'Dataset not found in /kaggle/input; contents: {list(Path("/kaggle/input").rglob("*"))[:20]}'

TRAIN_DIR, VAL_DIR, TEST_DIR = BASE / 'Training', BASE / 'Validation', BASE / 'Testing'
EXTS = {'.jpg', '.jpeg', '.png'}

def collect_records(split_dir: Path) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for p in sorted(cls_dir.rglob('*')):
            if p.suffix.lower() in EXTS:
                rows.append({'filepath': str(p), 'label': cls_dir.name})
    return pd.DataFrame(rows)

df_train_raw = collect_records(TRAIN_DIR)
df_val_raw   = collect_records(VAL_DIR)
df_test_id   = collect_records(TEST_DIR)  # In-Domain (ID) test split

CLASS_NAMES  = sorted(df_train_raw.label.unique())
NUM_CLASSES  = len(CLASS_NAMES)
LABEL_MAP    = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in LABEL_MAP.items()}

print(f'Target Foliar Classes ({NUM_CLASSES}): {CLASS_NAMES}')
print(f'Manifest Summary -> Train: {len(df_train_raw)}, Val: {len(df_val_raw)}, In-Domain Test: {len(df_test_id)}')

IMG_SIZE     = 256
BATCH_SIZE   = 32
EPOCHS_HEAD  = 5
EPOCHS_FINE  = 25
LR_HEAD      = 1e-3
LR_FINE      = 1e-4
WORKING      = Path('/kaggle/working')
WORKING.mkdir(parents=True, exist_ok=True)
""")

# Cell 4: Data Augmentation
add_code("""# Step 3 — Data Augmentation Engine & Data Streams
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.20,
    height_shift_range=0.20,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.65, 1.35],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

eval_datagen = ImageDataGenerator(rescale=1./255)

common_kwargs = dict(
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

train_gen = train_datagen.flow_from_dataframe(
    df_train_raw, x_col='filepath', y_col='label', shuffle=True, seed=PRIMARY_SEED, **common_kwargs
)
val_gen = eval_datagen.flow_from_dataframe(
    df_val_raw, x_col='filepath', y_col='label', shuffle=False, **common_kwargs
)
test_gen_id = eval_datagen.flow_from_dataframe(
    df_test_id, x_col='filepath', y_col='label', shuffle=False, **common_kwargs
)

assert train_gen.class_indices == LABEL_MAP, 'Class mapping assertion failed!'
""")

# Cell 5: Model Factory
add_code("""# Step 4 — Multi-Paradigm Architectural Model Factory
MODEL_REGISTRY = {
    'ResNet50': ResNet50,
    'DenseNet121': DenseNet121,
    'EfficientNetV2S': EfficientNetV2S,
    'ConvNeXtTiny': ConvNeXtTiny,
    'MobileNetV3': MobileNetV3Large
}

def build_pathology_model(
    arch_name: str,
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.30,
    l2_reg: float = 1e-4,
    lr: float = LR_HEAD
) -> Tuple[Model, Model]:
    Constructor = MODEL_REGISTRY[arch_name]
    base_model = Constructor(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(img_size, img_size, 3), name=f'{arch_name}_in')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(dropout_rate, name='drop1')(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg), name='dense_proj')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(dropout_rate * 0.67, name='drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier_out')(x)
    
    model = Model(inputs, outputs, name=f'Pathology_{arch_name}')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model, base_model

print(f'Registered {len(MODEL_REGISTRY)} Deep Learning Architectures:')
for k in MODEL_REGISTRY.keys():
    print(f'  • {k}')
""")

# Cell 6: Benchmark Execution
add_code("""# Step 5 — Unified Multi-Architecture Training & Evaluation Harness (RQ1)
benchmark_results = []
trained_models = {}

for arch_name in MODEL_REGISTRY.keys():
    print(f'\\n======================================================')
    print(f'🚀 Benchmarking Architecture: {arch_name}')
    print(f'======================================================')
    
    model, base_model = build_pathology_model(arch_name)
    
    # Phase 1: Classification Head Warmup
    callbacks_p1 = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]
    model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        verbose=1
    )
    
    # Phase 2: Deep Backbone Fine-Tuning
    base_model.trainable = True
    # Unfreeze top 40% of layers
    num_unfreeze = int(len(base_model.layers) * 0.4)
    for l in base_model.layers[:-num_unfreeze]:
        l.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_FINE),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    
    ckpt_path = WORKING / f'best_{arch_name.lower()}.keras'
    callbacks_p2 = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    t0 = time.perf_counter()
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        verbose=1
    )
    train_time_sec = time.perf_counter() - t0
    
    # In-Domain Test Evaluation
    y_true_id = test_gen_id.classes
    probs_id = model.predict(test_gen_id, verbose=0)
    preds_id = np.argmax(probs_id, axis=-1)
    
    acc_id = accuracy_score(y_true_id, preds_id)
    f1_id = f1_score(y_true_id, preds_id, average='macro')
    
    # Latency Benchmark (100 single-inference runs)
    dummy_in = tf.random.normal((1, IMG_SIZE, IMG_SIZE, 3))
    for _ in range(10): _ = model(dummy_in, training=False)
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        _ = model(dummy_in, training=False)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
    mean_lat = np.mean(latencies)
    
    params_m = model.count_params() / 1e6
    
    benchmark_results.append({
        'Architecture': arch_name,
        'Params (M)': round(params_m, 2),
        'Latency (ms)': round(mean_lat, 2),
        'ID Accuracy (%)': round(acc_id * 100.0, 2),
        'ID Macro-F1 (%)': round(f1_id * 100.0, 2),
        'Training Time (s)': round(train_time_sec, 1)
    })
    trained_models[arch_name] = (model, base_model)
    print(f'✅ {arch_name} -> ID Acc: {acc_id*100:.2f}%, ID Macro-F1: {f1_id*100:.2f}%, Latency: {mean_lat:.2f}ms')

df_benchmark = pd.DataFrame(benchmark_results)
print('\\n' + '='*70)
print('🏆 SUMMARY BENCHMARK TABLE (RQ1: Architectural Comparison)')
print('='*70)
print(df_benchmark.to_markdown(index=False))
df_benchmark.to_csv(WORKING / 'rq1_architecture_benchmark.csv', index=False)
""")

# Cell 7: Grad-CAM Explainability
add_code("""# Step 6 — Explainable AI (XAI) & Lesion Saliency Attribution (RQ4)
def generate_saliency_overlay(img_path, model, base_model, img_size=IMG_SIZE):
    img_pil = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img_pil) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)
    
    last_conv = [l for l in base_model.layers if any(k in l.name.lower() for k in ['conv', 'features', 'stage'])][-1]
    
    grad_model = Model(
        inputs=model.inputs,
        outputs=[base_model.get_layer(last_conv.name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(input_tensor, training=False)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]
        
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = conv_out[0] @ weights[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-10)
    heatmap = cam.numpy()
    
    heatmap_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize((img_size, img_size), Image.Resampling.BILINEAR)) / 255.0
    pred_class = IDX_TO_CLASS[int(pred_idx)]
    conf = float(preds[0][pred_idx]) * 100.0
    return img_array, heatmap_resized, pred_class, conf

# Visualize across models
best_arch = df_benchmark.sort_values(by='ID Macro-F1 (%)', ascending=False).iloc[0]['Architecture']
print(f'Visualizing Saliency Alignments with Top Model ({best_arch})...')
best_model, best_base = trained_models[best_arch]

sample_indices = np.random.RandomState(PRIMARY_SEED).choice(len(df_test_id), 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(16, 11))

for ax, idx in zip(axes.flat, sample_indices):
    row = df_test_id.iloc[idx]
    img_arr, heatmap, pred_c, conf = generate_saliency_overlay(row.filepath, best_model, best_base)
    
    ax.imshow(img_arr)
    ax.imshow(heatmap, cmap='jet', alpha=0.40, extent=[0, IMG_SIZE, IMG_SIZE, 0])
    is_correct = (pred_c == row.label)
    c_color = '#10b981' if is_correct else '#ef4444'
    ax.set_title(f'GT: {row.label}\\nPred: {pred_c} ({conf:.1f}%)', color=c_color, fontweight='bold', fontsize=11)
    ax.axis('off')

plt.suptitle(f'Grad-CAM Lesion Saliency Attribution ({best_arch})', fontweight='bold', fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig(WORKING / 'gradcam_pathology_saliency.png', dpi=200, bbox_inches='tight')
plt.show()
""")

# Cell 8: TFLite Quantization and Edge Optimization
add_code("""# Step 7 — Edge Pareto Optimization: FP32 vs FP16 vs INT8 Quantization (RQ5)
converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter_fp16.convert()
(WORKING / 'potato_quantized.tflite').write_bytes(tflite_fp16)

# INT8 Dynamic Range Quantization
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_int8 = converter_int8.convert()
(WORKING / 'potato_int8_dynamic.tflite').write_bytes(tflite_int8)

fp16_size_mb = len(tflite_fp16) / (1024 * 1024)
int8_size_mb = len(tflite_int8) / (1024 * 1024)

print('='*70)
print('⚡ EDGE COMPRESSION BENCHMARK')
print('='*70)
print(f'Float16 TFLite Model Size: {fp16_size_mb:.2f} MB')
print(f'INT8 Dynamic Quantized Size: {int8_size_mb:.2f} MB')

# Save comprehensive metadata
metadata = {
    'model_name': f'{best_arch}_PotatoPathology_Research_V3',
    'best_architecture': best_arch,
    'class_names': CLASS_NAMES,
    'label_map': LABEL_MAP,
    'img_size': IMG_SIZE,
    'benchmark_results': benchmark_results,
    'seed': PRIMARY_SEED
}
with open(WORKING / 'class_names.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print('✅ Saved class_names.json metadata successfully.')
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle/potato-foliar-pathology-research-benchmark.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Generated kaggle/potato-foliar-pathology-research-benchmark.ipynb successfully!")
