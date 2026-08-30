"""
Explainable AI (XAI) Engine: Grad-CAM and Grad-CAM++ for Foliar Pathology.
Enables visual localization of necrotic lesions, chlorotic halos, and background shortcut analysis.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model


def get_last_conv_layer_name(model: Model) -> str:
    """
    Heuristically finds the target convolutional layer for gradient extraction.
    """
    for layer in reversed(model.layers):
        # Check if sub-model (backbone) or single layer
        if isinstance(layer, Model):
            for sub_layer in reversed(layer.layers):
                if any(k in sub_layer.name.lower() for k in ["conv", "top_conv", "features", "stage"]):
                    return sub_layer.name
        if any(k in layer.name.lower() for k in ["conv", "top_conv", "features", "stage"]):
            return layer.name
    raise ValueError("Could not find a convolutional layer in the given model.")


def compute_gradcam(
    model: Model,
    base_model: Model,
    img_array: np.ndarray,
    target_class_idx: Optional[int] = None,
    use_plus_plus: bool = False,
) -> np.ndarray:
    """
    Computes Grad-CAM or Grad-CAM++ heatmap.
    
    Args:
        model: Full pathology classifier.
        base_model: Extractor backbone.
        img_array: Preprocessed image of shape (1, H, W, 3).
        target_class_idx: Class index for saliency (default: predicted class).
        use_plus_plus: If True, uses Grad-CAM++ formulation with second-order gradients.
    """
    last_conv_name = get_last_conv_layer_name(base_model)
    last_conv_layer = base_model.get_layer(last_conv_name)
    
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, target_class_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    
    if not use_plus_plus:
        # Standard Grad-CAM
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = conv_outputs[0] @ weights[..., tf.newaxis]
    else:
        # Grad-CAM++: Higher-order weighting for multiple lesion instances
        grads_power_2 = tf.square(grads)
        grads_power_3 = grads_power_2 * grads
        sum_conv = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        
        aij = grads_power_2 / (2.0 * grads_power_2 + sum_conv * grads_power_3 + 1e-10)
        aij = tf.where(grads != 0, aij, tf.zeros_like(aij))
        
        weights = tf.reduce_sum(aij * tf.maximum(grads, 0), axis=(1, 2))
        cam = tf.reduce_sum(conv_outputs[0] * weights[0], axis=-1, keepdims=True)
        
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-10)
    return cam.numpy()


def overlay_saliency_map(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
    colormap: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes heatmap and blends with original image.
    Returns:
        (blended_image_array, resized_heatmap_array)
    """
    import matplotlib.pyplot as plt
    
    img_w, img_h = original_img.size
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((img_w, img_h), Image.Resampling.BILINEAR)
    ) / 255.0
    
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]
    
    orig_arr = np.array(original_img) / 255.0
    blended = alpha * colored_heatmap + (1.0 - alpha) * orig_arr
    blended = np.clip(blended, 0.0, 1.0)
    
    return blended, heatmap_resized
