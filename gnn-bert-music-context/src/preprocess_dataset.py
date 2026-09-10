"""
Fills in the real-dataset stub described in datasets.py / README:
walks data/raw/<name>/, extracts features + builds a graph per clip
(audio_features.py + graph_builder.py), and writes one .pt file per
clip into data/processed/<name>/{train,val,test}/, plus:
  - data/splits/<dataset>_splits.json  (train/val/test track_id manifest)
  - data/processed/<dataset>/tag_names.json  (tag vocabulary, for Task 4
    zero-shot tag prediction prompts)

Worked example: GTZAN (10 genre folders, spec's "Easy baseline").
The same pattern applies to FMA/MagnaTagATune/MusicCaps/DEAM — only the
label/caption extraction (genre_to_multihot / genre_to_caption_tokens)
changes; audio -> graph is dataset-agnostic.

Usage:
    python src/preprocess_dataset.py --dataset gtzan \
        --raw-root data/raw/gtzan --out-root data/processed/gtzan
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio_features import load_audio, extract_segment_features
from src.graph_builder import build_segment_graph
from src.datasets import tokenize  # swap for a real HF tokenizer once using HFBertEncoder
GTZAN_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"]
GTZAN_GENRE_WORDS = {
    "blues": ["mournful", "soulful", "slide-guitar", "twelve-bar", "weary", "harmonica"],
    "classical": ["orchestral", "symphonic", "stately", "ornate", "composed", "refined"],
    "country": ["twangy", "storytelling", "rural", "banjo", "heartfelt", "dusty"],
    "disco": ["groovy", "four-on-the-floor", "glittery", "strings", "dancefloor", "retro"],
    "hiphop": ["rhythmic", "sampled", "spoken-word", "urban", "bassy", "looped"],
    "jazz": ["improvised", "syncopated", "smoky", "brassy", "swinging", "intricate"],
    "metal": ["distorted", "heavy", "aggressive", "screaming", "powerful", "intense"],
    "pop": ["catchy", "polished", "bright", "radio-ready", "hooky", "upbeat"],
    "reggae": ["offbeat", "laid-back", "island", "skanking", "mellow", "loping"],
    "rock": ["driving", "electric", "riff-heavy", "energetic", "raw", "anthemic"],
}

def genre_to_multihot(genre: str) -> np.ndarray:
    vec = np.zeros(len(GTZAN_GENRE_WORDS), dtype=np.float32)
    vec[list(GTZAN_GENRE_WORDS.keys()).index(genre)] = 1.0
    return vec


def genre_to_caption_tokens(genre: str, max_length=32):
    """GTZAN has no real captions. Instead of stating the genre name
    directly (which would let the model win by memorizing one token per
    class — a leakage shortcut, not genuine understanding), we describe
    the genre's *style* with adjectives, the way a MusicCaps-style caption
    would. The model has to learn which stylistic words correlate with
    which genre, from a caption that never says the answer outright."""
    pool = GTZAN_GENRE_WORDS[genre]
    chosen = random.sample(pool, k=min(3, len(pool)))
    words = ["a", "track", "with"] + chosen + ["sound"]
    random.shuffle(words)
    return tokenize(words, max_length)


def process_one_file(path, genre, max_length=32, window_seconds=5):
    try:
        y, sr = load_audio(path)
    except Exception as e:
        print(f"[skip] could not read {path}: {e}")
        return None
    seg_feats = extract_segment_features(y, sr, window_seconds=window_seconds)
    if seg_feats.shape[0] < 2:
        return None  # too short to build a meaningful graph, skip
    node_feats, edge_index, edge_weight = build_segment_graph(seg_feats, tau=0.6)
    tags = genre_to_multihot(genre)
    caption_ids = np.array(genre_to_caption_tokens(genre, max_length), dtype=np.int64)
    return dict(
        track_id=os.path.splitext(os.path.basename(path))[0],
        concept=GTZAN_GENRES.index(genre),
        tags=torch.tensor(tags),
        caption_ids=torch.tensor(caption_ids),
        node_feats=torch.tensor(node_feats),
        edge_index=torch.tensor(edge_index),
        # GTZAN has no valence/arousal labels; use 0.0 placeholders so the
        # Task 3 multitask loss still runs (its DEAM-only auxiliary head is
        # optional in practice — mask it out in training if you don't want
        # GTZAN to supervise emotion at all).
        valence=0.0,
        arousal=0.0,
        mel=torch.tensor(node_feats.mean(axis=0)),  # pooled descriptor, matches synthetic 'mel' shape
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gtzan")
    p.add_argument("--raw-root", default="data/raw/gtzan")
    p.add_argument("--out-root", default="data/processed/gtzan")
    p.add_argument("--splits-root", default="data/splits")
    p.add_argument("--window-seconds", type=int, default=5)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    files_by_genre = {}
    for genre in GTZAN_GENRES:
        files = sorted(glob.glob(os.path.join(args.raw_root, genre, "*.wav")))
        if not files:
            print(f"[warn] no .wav files found under {args.raw_root}/{genre} — skipping")
            continue
        files_by_genre[genre] = files

    if not files_by_genre:
        raise SystemExit(f"No audio found under {args.raw_root}. "
                          f"Check your GTZAN download layout (see Step 1 in the README).")

    n_ok, n_skipped = 0, 0
    split_manifest = {"train": [], "val": [], "test": []}  # track_ids per split, for reproducibility

    for genre, files in files_by_genre.items():
        random.shuffle(files)  # shuffle within genre before splitting -> stratified split
        n = len(files)
        n_val = int(n * args.val_frac)
        n_test = int(n * args.test_frac)
        split_files = {
            "val": files[:n_val],
            "test": files[n_val:n_val + n_test],
            "train": files[n_val + n_test:],
        }
        for split, split_paths in split_files.items():
            out_dir = os.path.join(args.out_root, split)
            os.makedirs(out_dir, exist_ok=True)
            for path in split_paths:
                sample = process_one_file(path, genre, args.max_length, args.window_seconds)
                if sample is None:
                    n_skipped += 1
                    continue
                track_id = sample["track_id"]
                out_path = os.path.join(out_dir, f"{track_id}.pt")
                torch.save(sample, out_path)
                split_manifest[split].append({"track_id": track_id, "genre": genre})
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f"[preprocess] {n_ok} clips processed...")

    # train/val/test manifest (spec's data/splits/*.json requirement)
    os.makedirs(args.splits_root, exist_ok=True)
    manifest_path = os.path.join(args.splits_root, f"{args.dataset}_splits.json")
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    # tag name vocabulary, used by Task 4 zero-shot tag prediction to build
    # text prompts like "a song about jazz"
    tag_names_path = os.path.join(args.out_root, "tag_names.json")
    with open(tag_names_path, "w") as f:
        json.dump(GTZAN_GENRES, f, indent=2)

    print(f"[preprocess] done. {n_ok} clips written to {args.out_root}/{{train,val,test}}, "
          f"{n_skipped} skipped (too short).")
    print(f"[preprocess] split manifest -> {manifest_path} "
          f"(train={len(split_manifest['train'])}, val={len(split_manifest['val'])}, "
          f"test={len(split_manifest['test'])})")
    print(f"[preprocess] tag names -> {tag_names_path}")


if __name__ == "__main__":
    main()