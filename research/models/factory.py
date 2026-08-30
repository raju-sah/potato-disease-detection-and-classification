"""
Multi-Paradigm Model Architecture Factory for Foliar Pathology Benchmark.
Supports:
  1. ResNet-50 (Residual Baseline)
  2. DenseNet-121 (Dense Feature Reuse)
  3. EfficientNetV2-S (Compound-Scaled CNN)
  4. ConvNeXt-Tiny (Modernized 7x7 Depthwise ConvNet)
  5. Swin-T / MobileNetV3 (Lightweight & Transformer-inspired alternatives)
"""

from typing import Tuple, Optional
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


SUPPORTED_ARCHITECTURES = [
    "resnet50",
    "densenet121",
    "efficientnet_v2s",
    "convnext_tiny",
    "mobilenet_v3",
]


def build_model_backbone(
    arch_name: str,
    num_classes: int = 3,
    img_size: int = 256,
    dropout_rate: float = 0.30,
    l2_reg: float = 1e-4,
    pretrained: bool = True,
) -> Tuple[Model, Model]:
    """
    Constructs a pathology classifier with a standardized classification head.
    
    Returns:
        (full_model, backbone_model)
    """
    arch = arch_name.lower().strip()
    weights = "imagenet" if pretrained else None
    input_shape = (img_size, img_size, 3)
    
    if arch == "resnet50":
        base_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    elif arch == "densenet121":
        base_model = DenseNet121(include_top=False, weights=weights, input_shape=input_shape)
    elif arch in ["efficientnet_v2s", "efficientnetv2s", "effnetv2"]:
        base_model = EfficientNetV2S(include_top=False, weights=weights, input_shape=input_shape)
    elif arch in ["convnext_tiny", "convnext"]:
        base_model = ConvNeXtTiny(include_top=False, weights=weights, input_shape=input_shape)
    elif arch in ["mobilenet_v3", "mobilenetv3"]:
        base_model = MobileNetV3Large(include_top=False, weights=weights, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {SUPPORTED_ARCHITECTURES}")
        
    base_model.trainable = False
    
    # Standardized Diagnostic Projection Head
    inputs = keras.Input(shape=input_shape, name=f"{arch}_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name=f"{arch}_gap")(x)
    x = layers.BatchNormalization(name=f"{arch}_bn1")(x)
    x = layers.Dropout(dropout_rate, name=f"{arch}_drop1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name=f"{arch}_dense1",
    )(x)
    x = layers.BatchNormalization(name=f"{arch}_bn2")(x)
    x = layers.Dropout(dropout_rate * 0.67, name=f"{arch}_drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pathology_classifier")(x)
    
    model = Model(inputs, outputs, name=f"FoliarPathology_{arch}")
    return model, base_model
