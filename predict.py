"""
Run inference on an audio file using the trained bark/whine classifier.
Returns confidence scores for each class.
"""

import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from train_model import BarkClassifier


def predict(audio_path: str, model_path: str = "models/bark_classifier.pt"):
    """Predict bark/whine confidence for an audio file."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    labels = checkpoint["labels"]
    idx_to_label = {v: k for k, v in labels.items()}

    # Load model
    model = BarkClassifier(n_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)

    if sr != config["sample_rate"]:
        waveform = T.Resample(sr, config["sample_rate"])(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_len = config["sample_rate"] * config["max_duration_s"]
    if waveform.shape[1] < target_len:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :target_len]

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )
    mel = mel_transform(waveform)
    mel_db = T.AmplitudeToDB()(mel)
    mel_db = mel_db.unsqueeze(0)  # add batch dim

    # Predict
    with torch.no_grad():
        logits = model(mel_db)
        probs = torch.softmax(logits, dim=1).squeeze()

    results = {}
    for idx in range(len(labels)):
        label = idx_to_label[idx]
        confidence = probs[idx].item() * 100
        results[label] = confidence

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file> [model_path]")
        print("  Supports: .wav, .mp3, .mp4, .flac")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/bark_classifier.pt"

    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train a model first: python train_model.py")
        sys.exit(1)

    results = predict(audio_path, model_path)

    print(f"\nPrediction for: {audio_path}")
    print("-" * 40)
    for label, confidence in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(confidence / 2)
        print(f"  {label:10s} {confidence:5.1f}% {bar}")


if __name__ == "__main__":
    main()
