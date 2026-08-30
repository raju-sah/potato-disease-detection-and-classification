"""
Statistical and Diagnostic Evaluation Metrics for Foliar Pathology.
Includes Macro-F1, Expected Calibration Error (ECE), Confusion Matrix metrics,
and Inference Latency Benchmarking.
"""

import time
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    log_loss,
)
import tensorflow as tf


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) to measure confidence trustworthiness.
    
    References:
        Guo et al. (2017), 'On Calibration of Modern Neural Networks', ICML.
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(np.float64)
    
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin.astype(np.float64))
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_sizes.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            
    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str] = ["Early_Blight", "Healthy", "Late_Blight"],
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of academic evaluation metrics.
    """
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    ece, _, _, _ = compute_expected_calibration_error(y_pred_probs, y_true)
    cm = confusion_matrix(y_true, y_pred)
    
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ece": ece,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def benchmark_inference_speed(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    warmup_runs: int = 15,
    benchmark_runs: int = 100,
) -> Dict[str, float]:
    """
    Measures mean latency and throughput under single-sample inference conditions.
    """
    dummy_input = tf.random.normal((1, *input_shape))
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)
        
    latencies_ms = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        
    mean_lat = float(np.mean(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    
    return {
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "throughput_fps": round(throughput, 1),
    }
