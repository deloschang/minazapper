# Mina Dog Vocalization Classifier

TFLite audio classifier that detects dog barking, whining, and background noise. Designed to run on Raspberry Pi 4 with <25ms inference time.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg  # macOS
```

## Training Data

Place WAV files in `./training_data/` with this structure:

```
training_data/
├── bark/       ← .wav files of dog barking
├── whine/      ← .wav files of dog whining
└── negative/   ← .wav files of ambient noise, speech, TV, HVAC, etc.
```

Audio is automatically resampled to 16kHz mono and normalized to 1 second.

### Adding new training samples

1. Record or download audio clips
2. Convert to WAV if needed: `ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav`
3. Place in the appropriate `training_data/` subdirectory
4. Retrain: `python train.py`

### Downloading from Unifi Protect

```bash
# Download bark detection clips
python download_clips.py --local-ip 192.168.1.1 --days 90

# Extract audio from video clips
python extract_audio.py

# Label clips interactively
python label_audio.py
```

## Training

```bash
python train.py
```

Outputs:
- `mina_classifier.tflite` — exported model
- `labels.txt` — class labels
- `models/mina_classifier.keras` — Keras model
- `models/test_data.npz` — held-out test set

## Evaluation

```bash
python evaluate.py
```

Outputs confusion matrix, per-class precision/recall/F1, and inference time benchmark.

## Inference

```python
import numpy as np
from utils import compute_mfcc

# Load TFLite model
from tflite_runtime.interpreter import Interpreter
interpreter = Interpreter(model_path="mina_classifier.tflite")
interpreter.allocate_tensors()

input_idx = interpreter.get_input_details()[0]["index"]
output_idx = interpreter.get_output_details()[0]["index"]

# Predict on 1-second audio window
audio = ...  # float32, normalized to [-1, 1], shape (16000,)
mfcc = compute_mfcc(audio)  # shape (124, 40)
interpreter.set_tensor(input_idx, mfcc[np.newaxis, :, :, np.newaxis])
interpreter.invoke()
scores = interpreter.get_tensor(output_idx)[0]
# scores = [bark_confidence, whine_confidence, negative_confidence]
```

## Model Architecture

```
Conv2D(8, 3x3, relu) → MaxPool2D
Conv2D(16, 3x3, relu) → MaxPool2D
Conv2D(32, 3x3, relu) → GlobalAvgPool2D
Dense(64, relu) → Dropout(0.25)
Dense(3, softmax)
```

Input: (1, 124, 40, 1) — MFCC features from 1-second audio window
Output: (1, 3) — [bark, whine, negative] confidence scores
