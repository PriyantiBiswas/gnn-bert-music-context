"""
datasets.py
-----------
Loaders for the datasets in spec Table 1 (FMA, MagnaTagATune, GTZAN, DEAM,
MusicCaps, MSD+tagtraum, Lakh MIDI, EmoMusic), plus a synthetic generator so
every training script (`train.py --synthetic`) is runnable without any
downloads.

Real-dataset loaders expect files already downloaded into `data/raw/<name>/`
per the official links in the project spec, and read cached
graphs/features/labels from `data/processed/` (built via
`audio_features.py` + `graph_builder.py`). They are written as thin,
well-documented stubs: fill in the paths for your actual download layout.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio_features import synthetic_track, segment_windows, segment_embedding
from graph_builder import (
    estimate_chord_sequence,
    build_chord_transition_graph,
    build_segment_similarity_graph,
)

TOP_50_MAGNATAGATUNE_TAGS = [
    "guitar", "classical", "slow", "techno", "strings", "drums", "electronic",
    "rock", "fast", "piano", "ambient", "beat", "violin", "vocal", "synth",
    "female", "indian", "opera", "male", "singing", "vocals", "no vocals",
    "harpsichord", "loud", "quiet", "flute", "woman", "male vocal",
    "no vocal", "pop", "soft", "sitar", "solo", "man", "classic", "choir",
    "voice", "new age", "dance", "male voice", "female vocal", "beats",
    "harp", "cello", "no voice", "weird", "country", "metal", "female voice",
    "choral", "jazz",
]

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop", "jazz",
    "metal", "pop", "reggae", "rock",
]


@dataclass
class MusicSample:
    track_id: str
    text: str                    # tags / caption / lyrics (raw string, joined tags OK)
    tags: np.ndarray              # multi-hot (K,)
    graph: object                 # Data / SimpleGraph from graph_builder
    genre: int | None = None
    valence: float | None = None
    arousal: float | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Real-dataset stubs (fill in paths once data is downloaded per Table 1)
# ---------------------------------------------------------------------------
def load_fma(root: str = "data/raw/fma_small", split: str = "train") -> list[MusicSample]:
    """FMA small/medium: genre + tags + metadata. See https://github.com/mdeff/fma"""
    root = Path(root)
    meta_path = root / "tracks.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Download FMA per Table 1 and place under {root}, "
            f"or run with --synthetic for a smoke test."
        )
    raise NotImplementedError(
        "Populate this loader once FMA is downloaded: parse tracks.csv, "
        "load cached mel/chroma from data/processed/fma/, build segment graphs "
        "via graph_builder.build_segment_similarity_graph, and apply the official "
        "FMA train/val/test split (no artist leakage)."
    )


def load_magnatagatune(root: str = "data/raw/magnatagatune") -> list[MusicSample]:
    """MagnaTagATune: 188 multi-label tags, 25,877 clips -> restrict to top-50 tags."""
    root = Path(root)
    annot_path = root / "annotations_final.csv"
    if not annot_path.exists():
        raise FileNotFoundError(
            f"{annot_path} not found. Download MagnaTagATune per Table 1, "
            f"or run with --synthetic for a smoke test."
        )
    raise NotImplementedError(
        "Populate once downloaded: filter to TOP_50_MAGNATAGATUNE_TAGS, "
        "tokenize tag string with BERT tokenizer, build segment graphs from clip audio."
    )


def load_musiccaps(root: str = "data/raw/musiccaps") -> list[MusicSample]:
    """MusicCaps: 5,521 clips with natural-language captions (Google)."""
    root = Path(root)
    jsonl_path = root / "musiccaps.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} not found. Download MusicCaps per Table 1, "
            f"or run with --synthetic for a smoke test."
        )
    raise NotImplementedError(
        "Populate once downloaded: read caption + ytid rows, pull the "
        "corresponding 10s audio clip, build a segment graph per clip for "
        "Task 4 contrastive (graph, caption) pairs."
    )


def load_deam(root: str = "data/raw/deam") -> list[MusicSample]:
    """DEAM: continuous valence/arousal (1-9) every 0.5s -> use song-level mean as target."""
    root = Path(root)
    annot_path = root / "annotations" / "song_level.csv"
    if not annot_path.exists():
        raise FileNotFoundError(
            f"{annot_path} not found. Download DEAM per Table 1, "
            f"or run with --synthetic for a smoke test."
        )
    raise NotImplementedError(
        "Populate once downloaded: aggregate per-timestep valence/arousal to "
        "song-level mean, attach as auxiliary regression target for Task 3."
    )


# ---------------------------------------------------------------------------
# Synthetic dataset (default for --synthetic; also used by unit tests)
# ---------------------------------------------------------------------------
def make_synthetic_dataset(
    n_samples: int = 200,
    num_tags: int = 50,
    num_genres: int = 8,
    graph_type: str = "segment",   # "segment" | "chord"
    seed: int = 42,
) -> list[MusicSample]:
    """
    Generates a small, internally-consistent synthetic dataset:
    each sample gets a random genre, a text description drawn from a
    genre-correlated tag vocabulary, a structure graph, and valence/arousal
    targets loosely correlated with the genre — enough signal for a model to
    learn something better than random, while requiring no downloads.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    genre_names = GTZAN_GENRES[:num_genres]
    tag_vocab = TOP_50_MAGNATAGATUNE_TAGS[:num_tags]
    # correlate a handful of tags with each genre so the task is learnable
    genre_tag_bias = {
        g: rng.sample(tag_vocab, k=min(5, len(tag_vocab))) for g in genre_names
    }
    genre_valence = {g: np_rng.uniform(2, 8) for g in genre_names}
    genre_arousal = {g: np_rng.uniform(2, 8) for g in genre_names}
    # Give each genre a fixed random "audio fingerprint" so that graph/audio
    # features (not just text) actually carry genre signal -- otherwise a
    # GNN trained on audio-only features has nothing to learn from.
    node_feat_dim = 25  # 12 mean + 12 std (chroma) + 1 (chord one-hot minor bit, unused here)
    genre_audio_bias = {
        g: np_rng.normal(0, 1.5, size=node_feat_dim).astype(np.float32) for g in genre_names
    }

    samples = []
    for i in range(n_samples):
        genre_idx = i % num_genres
        genre = genre_names[genre_idx]

        track = synthetic_track(seed=seed + i)
        if graph_type == "chord":
            chords = estimate_chord_sequence(track["chroma"])
            graph = build_chord_transition_graph(chords)
            bias = genre_audio_bias[genre][: graph.x.shape[1]]
            graph.x = graph.x + bias[None, :]
        else:
            segs = segment_windows(track["chroma"], sr=track["sr"], window_seconds=8.0)
            embeds = [segment_embedding(s) for s in segs]
            if len(embeds) < 2:
                embeds = embeds * 2
            bias = genre_audio_bias[genre][: embeds[0].shape[0]]
            embeds = [e + bias for e in embeds]
            graph = build_segment_similarity_graph(embeds, tau=0.85)

        active_tags = set(genre_tag_bias[genre])
        active_tags |= set(rng.sample(tag_vocab, k=2))  # noise tags
        tags = np.zeros(num_tags, dtype=np.float32)
        for t in active_tags:
            tags[tag_vocab.index(t)] = 1.0

        text = f"{genre} track featuring " + ", ".join(sorted(active_tags))

        samples.append(
            MusicSample(
                track_id=f"synth_{i:04d}",
                text=text,
                tags=tags,
                graph=graph,
                genre=genre_idx,
                valence=float(np_rng.normal(genre_valence[genre], 0.5)),
                arousal=float(np_rng.normal(genre_arousal[genre], 0.5)),
            )
        )
    return samples


def train_val_test_split(
    samples: list[MusicSample], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[MusicSample]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return {
        "train": [samples[i] for i in idx[:n_train]],
        "val": [samples[i] for i in idx[n_train : n_train + n_val]],
        "test": [samples[i] for i in idx[n_train + n_val :]],
    }


def save_split_ids(splits: dict[str, list[MusicSample]], out_path: str) -> None:
    payload = {k: [s.track_id for s in v] for k, v in splits.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    ds = make_synthetic_dataset(n_samples=20)
    splits = train_val_test_split(ds)
    save_split_ids(splits, "data/splits/synthetic_split.json")
    print({k: len(v) for k, v in splits.items()})
    print("example text:", ds[0].text)
    print("example graph:", ds[0].graph)
