"""
Preprocesses DEAM for Task 3 (real emotion-regression auxiliary loss) and
Task 4 (contrastive retrieval with genuinely varying captions).

DEAM has no genre/tag labels and no real captions/lyrics. Rather than use
a dummy constant tag and a dummy constant caption (which would make Task 3's
tag metrics meaningless and make Task 4 impossible -- every caption would
be identical, so there is no "own caption vs. other captions" to align),
this script derives BOTH from the real, continuous valence/arousal values:

  - tags: 4-class mood quadrant (happy/energetic, calm/content,
    angry/tense, sad/melancholic), binned from real (valence, arousal)
    around the midpoint of the 1-9 scale. This is the same quadrant
    scheme the spec's EmoMusic dataset uses, applied here to DEAM's
    continuous scores instead of pre-binned labels.
  - captions: descriptive adjectives for that clip's quadrant (not the
    literal quadrant name), so captions genuinely vary across clips and
    correlate with real emotional content -- giving Task 4 a real
    alignment problem to solve, and Task 3 a meaningful tag target
    alongside the real valence/arousal regression target.

Usage:
    python src/preprocess_deam.py \
        --audio-root data/raw/deam/audio \
        --annotations data/raw/deam/static_annotations.csv \
        --out-root data/processed/deam
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio_features import load_audio, extract_segment_features
from src.graph_builder import build_segment_graph
from src.datasets import tokenize

MOOD_QUADRANTS = ["happy_energetic", "calm_content", "angry_tense", "sad_melancholic"]

QUADRANT_WORDS = {
    "happy_energetic": ["upbeat", "joyful", "vibrant", "triumphant", "sparkling", "exuberant"],
    "calm_content": ["peaceful", "warm", "gentle", "soothing", "serene", "tender"],
    "angry_tense": ["aggressive", "tense", "frantic", "harsh", "chaotic", "jarring"],
    "sad_melancholic": ["melancholic", "somber", "mournful", "bleak", "weary", "hollow"],
}


def valence_arousal_to_quadrant(valence, arousal, mid=5.0):
    """DEAM's static annotations are on a 1-9 scale; midpoint 5.0 splits
    each axis into high/low, giving the 4 standard mood quadrants."""
    if valence >= mid and arousal >= mid:
        return "happy_energetic"
    if valence >= mid and arousal < mid:
        return "calm_content"
    if valence < mid and arousal >= mid:
        return "angry_tense"
    return "sad_melancholic"


def quadrant_to_multihot(quadrant: str) -> np.ndarray:
    vec = np.zeros(len(MOOD_QUADRANTS), dtype=np.float32)
    vec[MOOD_QUADRANTS.index(quadrant)] = 1.0
    return vec


def quadrant_to_caption_tokens(quadrant: str, max_length=32):
    pool = QUADRANT_WORDS[quadrant]
    chosen = random.sample(pool, k=min(3, len(pool)))
    words = ["a", "track", "with"] + chosen + ["mood"]
    random.shuffle(words)
    return tokenize(words, max_length)


def load_annotations(path):
    """Returns {song_id (str): (valence, arousal)}. `path` may be a single
    CSV file, or a folder containing one or more CSVs (DEAM often ships
    annotations split into 2 files, e.g. songs 1-2000 and 2000-2058) --
    every CSV found is read and merged. DEAM's column names have also
    varied slightly across releases (e.g. ' valence_mean' with a leading
    space) -- this strips whitespace and looks for the first column
    containing 'valence'/'arousal' (excluding *_std columns)."""
    if os.path.isdir(path):
        csv_paths = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not csv_paths:
            raise SystemExit(f"No .csv files found in folder {path}")
    else:
        csv_paths = [path]

    out = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        id_col = next(c for c in df.columns if c.lower() in ("song_id", "songid", "id"))
        val_col = next(c for c in df.columns if "valence" in c.lower() and "std" not in c.lower())
        aro_col = next(c for c in df.columns if "arousal" in c.lower() and "std" not in c.lower())
        for _, row in df.iterrows():
            out[str(int(row[id_col]))] = (float(row[val_col]), float(row[aro_col]))
        print(f"[preprocess-deam] loaded {len(df)} annotation rows from {csv_path}")
    return out

def process_one_file(path, valence, arousal, max_length=32, window_seconds=5):
    try:
        y, sr = load_audio(path)
    except Exception as e:
        print(f"[skip] could not read {path}: {e}")
        return None
    seg_feats = extract_segment_features(y, sr, window_seconds=window_seconds)
    if seg_feats.shape[0] < 2:
        return None
    node_feats, edge_index, edge_weight = build_segment_graph(seg_feats, tau=0.6)

    quadrant = valence_arousal_to_quadrant(valence, arousal)
    tags = quadrant_to_multihot(quadrant)
    caption_ids = np.array(quadrant_to_caption_tokens(quadrant, max_length), dtype=np.int64)

    return dict(
        track_id=os.path.splitext(os.path.basename(path))[0],
        concept=MOOD_QUADRANTS.index(quadrant),
        tags=torch.tensor(tags),
        caption_ids=torch.tensor(caption_ids),
        node_feats=torch.tensor(node_feats),
        edge_index=torch.tensor(edge_index),
        valence=float(valence),
        arousal=float(arousal),
        mel=torch.tensor(node_feats.mean(axis=0)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-root", default="data/raw/deam/audio")
    p.add_argument("--annotations", default="data/raw/deam/static_annotations.csv")
    p.add_argument("--out-root", default="data/processed/deam")
    p.add_argument("--splits-root", default="data/splits")
    p.add_argument("--window-seconds", type=int, default=5)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    anno = load_annotations(args.annotations)
    files = sorted(glob.glob(os.path.join(args.audio_root, "*.wav")) +
                   glob.glob(os.path.join(args.audio_root, "*.mp3")))
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in anno]
    if not files:
        raise SystemExit(f"No audio files under {args.audio_root} matched an entry in "
                          f"{args.annotations}. Check filenames match song_id exactly "
                          f"(e.g. '2.wav' should match song_id=2 in the CSV).")

    random.shuffle(files)
    n = len(files)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    split_files = {
        "val": files[:n_val],
        "test": files[n_val:n_val + n_test],
        "train": files[n_val + n_test:],
    }

    n_ok, n_skipped = 0, 0
    quadrant_counts = {q: 0 for q in MOOD_QUADRANTS}
    split_manifest = {"train": [], "val": [], "test": []}
    for split, split_paths in split_files.items():
        out_dir = os.path.join(args.out_root, split)
        os.makedirs(out_dir, exist_ok=True)
        for path in split_paths:
            song_id = os.path.splitext(os.path.basename(path))[0]
            valence, arousal = anno[song_id]
            sample = process_one_file(path, valence, arousal, args.max_length, args.window_seconds)
            if sample is None:
                n_skipped += 1
                continue
            out_path = os.path.join(out_dir, f"{song_id}.pt")
            torch.save(sample, out_path)
            quadrant = MOOD_QUADRANTS[sample["concept"]]
            quadrant_counts[quadrant] += 1
            split_manifest[split].append({
                "track_id": song_id, "valence": valence, "arousal": arousal, "quadrant": quadrant,
            })
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"[preprocess-deam] {n_ok} clips processed...")

    os.makedirs(args.splits_root, exist_ok=True)
    manifest_path = os.path.join(args.splits_root, "deam_splits.json")
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    tag_names_path = os.path.join(args.out_root, "tag_names.json")
    with open(tag_names_path, "w") as f:
        json.dump(MOOD_QUADRANTS, f, indent=2)

    print(f"[preprocess-deam] done. {n_ok} clips written to {args.out_root}/{{train,val,test}}, "
          f"{n_skipped} skipped.")
    print(f"[preprocess-deam] split manifest -> {manifest_path} "
          f"(train={len(split_manifest['train'])}, val={len(split_manifest['val'])}, "
          f"test={len(split_manifest['test'])})")
    print(f"[preprocess-deam] mood quadrant distribution: {quadrant_counts}")
    if min(quadrant_counts.values(), default=0) == 0:
        print("[preprocess-deam] WARNING: at least one quadrant has zero clips -- "
              "your DEAM sample may be skewed toward one mood region; check "
              "quadrant_counts above before training Task 3/4 on this data.")


if __name__ == "__main__":
    main()