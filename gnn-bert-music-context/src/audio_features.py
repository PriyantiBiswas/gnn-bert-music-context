"""
audio_features.py
------------------
Audio preprocessing pipeline (spec Section 3):
  1. Resample to 22,050 Hz
  2. Extract log-mel spectrogram (128 bins) or chroma (12 bins), normalize per track
  3. Segment into fixed windows (5-10s) or beat-synchronous segments (librosa)

Every function has a pure-numpy synthetic fallback (`synthetic=True`) so the
rest of the pipeline (graph_builder, gnn_model, train) can be developed and
smoke-tested without downloading any audio.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    _HAS_LIBROSA = False


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load and resample an audio file to `sr` Hz (mono)."""
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for load_audio(); pip install librosa")
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_log_mel(y: np.ndarray, sr: int = 22050, n_mels: int = 128) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, T). Per-track z-normalized."""
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for extract_log_mel()")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return _znorm(log_S)


def extract_chroma(y: np.ndarray, sr: int = 22050, n_chroma: int = 12) -> np.ndarray:
    """Chroma features (for chord-transition graphs), shape (n_chroma, T)."""
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for extract_chroma()")
    C = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma)
    return _znorm(C)


def segment_windows(
    feat: np.ndarray, sr: int, hop_length: int = 512, window_seconds: float = 8.0
) -> list[np.ndarray]:
    """Split a (n_features, T) feature matrix into fixed-length windows."""
    frames_per_window = int(window_seconds * sr / hop_length)
    T = feat.shape[1]
    segments = [
        feat[:, i : i + frames_per_window]
        for i in range(0, T, frames_per_window)
        if feat[:, i : i + frames_per_window].shape[1] > 0
    ]
    return segments


def segment_beat_synchronous(y: np.ndarray, sr: int, feat: np.ndarray) -> list[np.ndarray]:
    """Beat-synchronous segmentation using librosa beat tracking."""
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for segment_beat_synchronous()")
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_frames = np.unique(np.concatenate(([0], beat_frames, [feat.shape[1]])))
    return [feat[:, s:e] for s, e in zip(beat_frames[:-1], beat_frames[1:]) if e > s]


def segment_embedding(segment: np.ndarray) -> np.ndarray:
    """Collapse a (n_features, t) segment into a single embedding via mean+std pooling."""
    mean = segment.mean(axis=1)
    std = segment.std(axis=1)
    return np.concatenate([mean, std], axis=0)


def _znorm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# Synthetic fallback (no audio files / no librosa required)
# ---------------------------------------------------------------------------
def synthetic_track(
    n_mels: int = 128, n_chroma: int = 12, n_frames: int = 1300, seed: int | None = None
) -> dict:
    """Generate a fake but structurally plausible track (mel + chroma + duration)."""
    rng = np.random.default_rng(seed)
    mel = rng.normal(0, 1, size=(n_mels, n_frames)).astype(np.float32)
    # give chroma some smooth, chord-like structure (piecewise-constant + noise)
    n_chords = max(4, n_frames // 200)
    chord_bounds = np.sort(rng.choice(np.arange(1, n_frames), n_chords - 1, replace=False))
    chord_bounds = np.concatenate(([0], chord_bounds, [n_frames]))
    chroma = np.zeros((n_chroma, n_frames), dtype=np.float32)
    for s, e in zip(chord_bounds[:-1], chord_bounds[1:]):
        root = rng.integers(0, n_chroma)
        vec = np.zeros(n_chroma)
        vec[[root, (root + 4) % n_chroma, (root + 7) % n_chroma]] = 1.0  # triad
        chroma[:, s:e] = vec[:, None] + rng.normal(0, 0.05, size=(n_chroma, e - s))
    return {"mel": mel, "chroma": chroma, "sr": 22050, "n_frames": n_frames}


if __name__ == "__main__":
    track = synthetic_track(seed=0)
    segs = segment_windows(track["chroma"], sr=track["sr"], window_seconds=8.0)
    print(f"synthetic track: mel {track['mel'].shape}, chroma {track['chroma'].shape}, "
          f"{len(segs)} segments")
