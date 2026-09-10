"""
Audio preprocessing pipeline: resample -> log-mel/chroma -> segment.
Used by preprocess_dataset.py on real audio files.
"""
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def _require_librosa():
    if librosa is None:
        raise ImportError("librosa is required: pip install librosa")


def load_audio(path, sr=22050):
    _require_librosa()
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def log_mel_spectrogram(y, sr=22050, n_mels=128):
    _require_librosa()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return _normalize(log_S)


def chroma_features(y, sr=22050, n_chroma=12):
    _require_librosa()
    C = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma)
    return _normalize(C)


def _normalize(feat):
    mean, std = feat.mean(), feat.std() + 1e-8
    return (feat - mean) / std


def segment_windows(y, sr=22050, window_seconds=5, beat_synchronous=False):
    _require_librosa()
    if beat_synchronous:
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_samples = librosa.frames_to_samples(beat_frames)
        bounds = [0] + list(beat_samples) + [len(y)]
        return [y[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    win = int(window_seconds * sr)
    return [y[i:i + win] for i in range(0, len(y), win) if len(y[i:i + win]) > sr // 2]


def extract_segment_features(y, sr=22050, window_seconds=5, n_mels=128):
    segments = segment_windows(y, sr, window_seconds)
    feats = []
    for seg in segments:
        if len(seg) < sr // 4:
            continue
        S = log_mel_spectrogram(seg, sr, n_mels)
        feats.append(S.mean(axis=1))
    return np.stack(feats) if feats else np.zeros((0, n_mels))