"""
Train a TFLite audio classifier for dog vocalization detection.

Classes: bark, whine, negative
Target: Raspberry Pi 4, <25ms inference on (1, 124, 40, 1) input
"""

import os
import random
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils import compute_mfcc, SAMPLE_RATE, WINDOW_SIZE

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Paths
TRAINING_DATA_DIR = Path("./training_data")
MODEL_OUTPUT = Path("mina_classifier.tflite")
LABELS_OUTPUT = Path("labels.txt")

# Labels in index order — must match model output
LABELS = ["mina", "negative"]


def load_audio(filepath: str) -> np.ndarray:
    """Load audio file, resample to 16kHz mono, normalize to 1 second."""
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    # Normalize to exactly 1 second
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
    else:
        audio = audio[:WINDOW_SIZE]
    return audio.astype(np.float32)


def augment_time_shift(audio: np.ndarray) -> np.ndarray:
    """Shift audio by ±10%."""
    shift = int(len(audio) * random.uniform(-0.1, 0.1))
    return np.roll(audio, shift)


def augment_gaussian_noise(audio: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, len(audio)).astype(np.float32)
    return audio + noise


def augment_pitch_shift(audio: np.ndarray) -> np.ndarray:
    """Pitch shift by ±2 semitones."""
    n_steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(
        audio, sr=SAMPLE_RATE, n_steps=n_steps
    ).astype(np.float32)


def load_dataset():
    """Load all training data and compute MFCCs."""
    X, y = [], []

    for label_idx, label_name in enumerate(LABELS):
        label_dir = TRAINING_DATA_DIR / label_name
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist, skipping.")
            continue

        wav_files = list(label_dir.glob("*.wav"))
        print(f"Loading {label_name}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                audio = load_audio(str(wav_path))
                mfcc = compute_mfcc(audio)
                X.append(mfcc)
                y.append(label_idx)
            except Exception as e:
                print(f"  Error loading {wav_path.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def augment_dataset(X_train, y_train):
    """Apply data augmentation to training set."""
    X_aug, y_aug = list(X_train), list(y_train)

    print(f"Augmenting {len(X_train)} training samples...")

    for i in range(len(X_train)):
        # We need to reconstruct audio from the original files for augmentation
        # Instead, we'll apply augmentation at the audio level during loading
        pass

    return np.array(X_aug), np.array(y_aug)


def load_dataset_with_augmentation():
    """Load dataset with augmentation applied to training split."""
    all_audio = []
    all_labels = []

    for label_idx, label_name in enumerate(LABELS):
        label_dir = TRAINING_DATA_DIR / label_name
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist, skipping.")
            continue

        wav_files = list(label_dir.glob("*.wav"))
        print(f"Loading {label_name}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                audio = load_audio(str(wav_path))
                all_audio.append(audio)
                all_labels.append(label_idx)
            except Exception as e:
                print(f"  Error loading {wav_path.name}: {e}")

    all_audio = np.array(all_audio, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    # Split: 70/15/15
    X_train_audio, X_temp_audio, y_train, y_temp = train_test_split(
        all_audio, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels
    )
    X_val_audio, X_test_audio, y_val, y_test = train_test_split(
        X_temp_audio, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"\nSplit sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # Augment training data at audio level
    print("Augmenting training data...")
    X_train_augmented = list(X_train_audio)
    y_train_augmented = list(y_train)

    for i in range(len(X_train_audio)):
        audio = X_train_audio[i]

        # Time shift
        shifted = augment_time_shift(audio)
        X_train_augmented.append(shifted)
        y_train_augmented.append(y_train[i])

        # Gaussian noise
        noisy = augment_gaussian_noise(audio)
        X_train_augmented.append(noisy)
        y_train_augmented.append(y_train[i])

        # Pitch shift
        pitched = augment_pitch_shift(audio)
        X_train_augmented.append(pitched)
        y_train_augmented.append(y_train[i])

    X_train_audio_aug = np.array(X_train_augmented, dtype=np.float32)
    y_train_aug = np.array(y_train_augmented, dtype=np.int32)

    print(f"After augmentation: {len(y_train_aug)} training samples "
          f"({len(y_train)} original + {len(y_train_aug) - len(y_train)} augmented)")

    # Convert all to MFCCs
    print("Computing MFCCs...")
    X_train = np.array([compute_mfcc(a) for a in X_train_audio_aug])
    X_val = np.array([compute_mfcc(a) for a in X_val_audio])
    X_test = np.array([compute_mfcc(a) for a in X_test_audio])

    # Add channel dimension: (N, 124, 40) → (N, 124, 40, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train_aug, X_val, y_val, X_test, y_test


def build_model(n_classes=2):
    """Build small CNN for Pi 4 inference."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(124, 40, 1)),
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    return model


def train():
    """Full training pipeline."""
    print("=" * 60)
    print("Mina Dog Vocalization Classifier — Training")
    print("=" * 60)

    # Load data with augmentation
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_with_augmentation()

    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # Compute class weights for imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # Build model
    model = build_model(n_classes=len(LABELS))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        ),
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save Keras model (for evaluate.py)
    keras_path = Path("models/mina_classifier.keras")
    keras_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(keras_path)
    print(f"Keras model saved to: {keras_path}")

    # Convert to TFLite (float32, no quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_OUTPUT, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {MODEL_OUTPUT}")
    print(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")

    # Save labels
    with open(LABELS_OUTPUT, "w") as f:
        for label in LABELS:
            f.write(label + "\n")
    print(f"Labels saved to: {LABELS_OUTPUT}")

    # Save test set for evaluation
    np.savez(
        "models/test_data.npz",
        X_test=X_test, y_test=y_test,
    )
    print("Test data saved to: models/test_data.npz")

    val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\nBest validation accuracy: {val_acc:.4f}")
    if val_acc >= 0.95:
        print("Target >95% validation accuracy: ACHIEVED")
    else:
        print("Target >95% validation accuracy: NOT YET — consider adding more training data")


if __name__ == "__main__":
    train()
