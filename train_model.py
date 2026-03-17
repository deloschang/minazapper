"""
Train a bark vs whine audio classifier.

Extracts mel-spectrogram features from labeled WAV files and trains
a small CNN that outputs confidence scores.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


# --- Config ---
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_DURATION_S = 10  # pad/crop to this length
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
MODEL_PATH = Path("models/bark_classifier.pt")

LABELS = {"barking": 0, "whining": 1}


class AudioDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.samples = []
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        for label_name, label_idx in LABELS.items():
            label_dir = data_dir / label_name
            if not label_dir.exists():
                continue
            for wav_path in label_dir.glob("*.wav"):
                self.samples.append((wav_path, label_idx))

        if not self.samples:
            raise ValueError(f"No samples found in {data_dir}. Expected barking/ and whining/ subdirs.")

        print(f"Loaded {len(self.samples)} samples:")
        for label_name in LABELS:
            count = sum(1 for _, l in self.samples if l == LABELS[label_name])
            print(f"  {label_name}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(str(path))

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or crop to fixed length
        target_len = SAMPLE_RATE * MAX_DURATION_S
        if waveform.shape[1] < target_len:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :target_len]

        # Convert to mel spectrogram
        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)

        return mel_db, label


class BarkClassifier(nn.Module):
    """Small CNN for bark/whine classification."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train():
    data_dir = Path("data")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = AudioDataset(data_dir)

    if len(dataset) < 10:
        print("Warning: Very few samples. Model quality will be limited.")
        print("Consider collecting more labeled data.")

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = BarkClassifier(n_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for mels, labels in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                outputs = model(mels)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / max(train_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "labels": LABELS,
                "config": {
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "max_duration_s": MAX_DURATION_S,
                },
            }, MODEL_PATH)
            print(f"  → Saved best model (val_acc={val_acc:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
