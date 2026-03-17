"""
Evaluate the trained Mina classifier.

Outputs confusion matrix, per-class precision/recall/F1,
and TFLite inference time benchmark.
"""

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ["mina", "negative"]
TFLITE_MODEL = Path("mina_classifier.tflite")
TEST_DATA = Path("models/test_data.npz")


def load_tflite_model(model_path):
    """Load TFLite model."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def run_evaluation():
    print("=" * 60)
    print("Mina Dog Vocalization Classifier — Evaluation")
    print("=" * 60)

    # Load test data
    data = np.load(TEST_DATA)
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"Test set: {len(y_test)} samples")

    # Load model
    interpreter = load_tflite_model(TFLITE_MODEL)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_idx = input_details[0]["index"]
    output_idx = output_details[0]["index"]

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Run predictions
    y_pred = []
    all_scores = []

    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_idx, sample)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_idx)[0]
        all_scores.append(scores)
        y_pred.append(np.argmax(scores))

    y_pred = np.array(y_pred)

    # Confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    # Header
    header = "          " + "  ".join(f"{l:>8s}" for l in LABELS)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"{LABELS[i]:>8s}  {row_str}")

    # Classification report
    print("\n--- Per-class Metrics ---")
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    print(report)

    # Inference time benchmark
    print("--- Inference Benchmark ---")
    # Create a dummy input matching expected shape
    dummy = np.random.randn(1, 124, 40, 1).astype(np.float32)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()

    # Benchmark: 1000 runs, single-threaded
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        interpreter.set_tensor(input_idx, dummy)
        interpreter.invoke()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    times = np.array(times)
    print(f"Inference time over 1000 runs:")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  P95:    {np.percentile(times, 95):.2f} ms")
    print(f"  P99:    {np.percentile(times, 99):.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")

    if np.median(times) < 25:
        print(f"\n  Target <25ms inference: PASS (median {np.median(times):.2f} ms)")
    else:
        print(f"\n  Target <25ms inference: FAIL (median {np.median(times):.2f} ms)")
        print("  Note: This benchmark is on your machine, not Pi 4.")
        print("  Pi 4 may be slower — consider quantization if needed.")

    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\nOverall test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    run_evaluation()
