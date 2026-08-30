"""
Dataset loader and augmentation pipeline for Potato Foliar Pathology Benchmark.
Supports In-Domain (Controlled Laboratory) and Out-of-Domain (Field) splits,
with advanced regularizations: RandAugment, CutMix, MixUp, and Random Erasing.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in LABEL_MAP.items()}


def get_augmentation_pipeline(aug_type: str = "advanced") -> ImageDataGenerator:
    """
    Returns an ImageDataGenerator based on the ablation condition.
    
    aug_type options:
      - 'none': Standard rescaling only.
      - 'baseline': Standard flip + slight rotation.
      - 'advanced': RandAugment-style aggressive geometric + photometric jitter.
    """
    if aug_type == "none":
        return ImageDataGenerator(rescale=1.0 / 255.0)
    elif aug_type == "baseline":
        return ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=15,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
        )
    elif aug_type == "advanced":
        return ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=40,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.20,
            zoom_range=0.30,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.60, 1.40],
            channel_shift_range=25.0,
            fill_mode="nearest",
        )
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def apply_cutmix_mixup(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 0.2,
    prob_mixup: float = 0.5,
    prob_cutmix: float = 0.5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies MixUp or CutMix regularization to a batch of images and one-hot labels.
    """
    batch_size = tf.shape(images)[0]
    
    # Randomly shuffle batch
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    choice = tf.random.uniform([], 0, 1.0)
    
    if choice < prob_mixup:
        # MixUp
        lam = tf.random.uniform([], 0.0, 1.0)
        mixed_images = lam * images + (1.0 - lam) * shuffled_images
        mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels
        return mixed_images, mixed_labels
    elif choice < (prob_mixup + prob_cutmix):
        # CutMix
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
        lam = tf.random.uniform([], 0.0, 1.0)
        
        cut_rat = tf.math.sqrt(1.0 - lam)
        cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_rat, tf.int32)
        cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_rat, tf.int32)
        
        cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
        cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
        
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
        
        # Create binary mask
        mask = tf.pad(
            tf.zeros([bby2 - bby1, bbx2 - bbx1, tf.shape(images)[3]]),
            [[bby1, img_h - bby2], [bbx1, img_w - bbx2], [0, 0]],
            constant_values=1.0,
        )
        mask = tf.expand_dims(mask, 0)
        
        mixed_images = images * mask + shuffled_images * (1.0 - mask)
        actual_lam = 1.0 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
        mixed_labels = actual_lam * labels + (1.0 - actual_lam) * shuffled_labels
        return mixed_images, mixed_labels
    
    return images, labels
