"""
Shared MFCC computation for training and inference.

CRITICAL: This exact function must be used in both training and runtime
to avoid training/inference mismatch.
"""

import numpy as np
import librosa


# Audio constants
SAMPLE_RATE = 16000
WINDOW_SIZE = 16000  # 1 second of audio
FRAME_LENGTH = 255
FRAME_STEP = 128
N_MELS = 40
N_MFCC = 40
# Output shape: (124, 40)


def compute_mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Compute MFCC features from a 1-second audio window.

    Args:
        audio: float32 array normalized to [-1, 1], shape (16000,)

    Returns:
        mfcc: float32 array, shape (124, 40)
    """
    audio = audio.astype(np.float32)

    # Ensure exactly 1 second
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
    elif len(audio) > WINDOW_SIZE:
        audio = audio[:WINDOW_SIZE]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH,
        hop_length=FRAME_STEP,
        n_mels=N_MELS,
        center=False,
    )

    # mfcc shape is (n_mfcc, time_steps) = (40, 124)
    # Transpose to (time_steps, n_mfcc) = (124, 40)
    mfcc = mfcc.T

    assert mfcc.shape == (124, 40), f"MFCC shape mismatch: {mfcc.shape}, expected (124, 40)"

    return mfcc.astype(np.float32)
