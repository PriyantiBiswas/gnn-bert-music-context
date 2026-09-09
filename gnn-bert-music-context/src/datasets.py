"""
Dataset loading for the GNN-BERT music context project.

Two modes:
  1. --synthetic : generates a small, *structured* synthetic corpus so the
     whole pipeline (BERT tagger, GNN, fusion, contrastive) can be trained
     and evaluated end-to-end with no downloads. The synthetic data is not
     random noise: each sample is drawn from one of a handful of latent
     "concepts" (proxy for genre/mood clusters) that jointly bias the tags,
     the caption tokens, the graph node features/edge density, and the
     valence/arousal targets. This gives every model something real to
     learn, so metrics move off the random baseline like they would on
     real data.
  2. real mode (--data-root / --dataset) : loaders that read *already
     preprocessed* .pt graph files written by src/preprocess_dataset.py
     under data/processed/<dataset>/{train,val,test}/.
"""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

CONCEPT_NAMES = [
    "melancholic_jazz", "high_energy_electronic", "acoustic_folk",
    "aggressive_rock", "ambient_chill", "upbeat_pop",
]
# Each concept "owns" a small cluster of tags out of a shared 20-tag vocab.
TAG_VOCAB = [
    "jazz", "melancholic", "slow_tempo", "piano",            # concept 0
    "electronic", "high_arousal", "synth", "danceable",      # concept 1
    "folk", "acoustic", "mellow", "guitar",                  # concept 2
    "rock", "distorted", "aggressive", "drums",               # concept 3
    "ambient", "calm", "atmospheric", "low_arousal",          # concept 4
    "pop", "catchy", "vocal", "bright",                       # concept 5
]
CONCEPT_TAG_SLICE = {i: slice(i * 4, i * 4 + 4) for i in range(6)}
CONCEPT_VA = {  # base (valence, arousal) in [1, 9] per concept, per Table 1 (DEAM range)
    0: (3.0, 3.5), 1: (6.5, 8.0), 2: (5.5, 4.0),
    3: (3.5, 8.0), 4: (5.0, 2.5), 5: (7.5, 6.5),
}
CONCEPT_WORDS = {
    0: ["mournful", "brushed", "cymbals", "lounge", "smoky", "minor"],
    1: ["pulsing", "four-on-the-floor", "sidechain", "rave", "neon", "drop"],
    2: ["fingerpicked", "campfire", "warm", "unplugged", "wooden", "porch"],
    3: ["crunchy", "riff", "screamed", "wall-of-sound", "power-chord", "grit"],
    4: ["floating", "drone", "reverb", "sparse", "night", "hush"],
    5: ["shiny", "hook", "chorus", "radio-ready", "clap", "sunny"],
}
COMMON_WORDS = ["the", "song", "features", "a", "track", "with", "sound", "of", "music"]


def _vocab():
    words = sorted(set(COMMON_WORDS + [w for ws in CONCEPT_WORDS.values() for w in ws]))
    return {w: i + 4 for i, w in enumerate(words)}  # 0..3 reserved (PAD,CLS,SEP,UNK)


VOCAB = _vocab()
PAD, CLS, SEP, UNK = 0, 1, 2, 3
VOCAB_SIZE = len(VOCAB) + 4


def tokenize(words, max_length=32):
    ids = [CLS] + [VOCAB.get(w, UNK) for w in words][: max_length - 2] + [SEP]
    ids = ids + [PAD] * (max_length - len(ids))
    return ids[:max_length]


def _make_graph(rng, concept, n_nodes=8, feat_dim=16):
    """Segment-similarity graph: node features cluster around a concept
    mean; edges = temporal chain + high-similarity pairs (mimics Section 3
    'segment graph': temporal adjacency + cosine similarity > tau)."""
    mean = rng.normal(loc=concept, scale=1.0, size=feat_dim) * 0.5
    feats = rng.normal(loc=mean, scale=0.6, size=(n_nodes, feat_dim)).astype(np.float32)
    edges = [(i, i + 1) for i in range(n_nodes - 1)]  # temporal chain
    edges += [(i + 1, i) for i in range(n_nodes - 1)]
    sims = feats @ feats.T / (np.linalg.norm(feats, axis=1, keepdims=True) *
                               np.linalg.norm(feats, axis=1, keepdims=True).T + 1e-8)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and sims[i, j] > 0.6:
                edges.append((i, j))
    edges = sorted(set(edges))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return feats, edge_index


class SyntheticMusicContextDataset(Dataset):
    """One synthetic dataset backing all four tasks (tags, graph, captions,
    valence/arousal), split via `split` ('train'/'val'/'test')."""

    def __init__(self, n_samples=480, split="train", seed=42, max_length=32):
        rng = np.random.default_rng(seed + hash(split) % 1000)
        self.max_length = max_length
        self.samples = []
        for i in range(n_samples):
            concept = int(rng.integers(0, len(CONCEPT_NAMES)))
            tags = np.zeros(len(TAG_VOCAB), dtype=np.float32)
            sl = CONCEPT_TAG_SLICE[concept]
            active = rng.choice(range(sl.start, sl.stop),
                                 size=rng.integers(2, 4), replace=False)
            tags[active] = 1.0
            # small cross-concept noise tag (models real-world label noise)
            if rng.random() < 0.1:
                tags[rng.integers(0, len(TAG_VOCAB))] = 1.0

            n_words = rng.integers(6, 14)
            words = list(rng.choice(CONCEPT_WORDS[concept], size=min(3, n_words), replace=False))
            words += list(rng.choice(COMMON_WORDS, size=n_words - len(words), replace=True))
            rng.shuffle(words)
            caption_ids = tokenize(words, max_length)

            feats, edge_index = _make_graph(rng, concept)

            va = CONCEPT_VA[concept]
            valence = float(np.clip(rng.normal(va[0], 0.6), 1, 9))
            arousal = float(np.clip(rng.normal(va[1], 0.6), 1, 9))

            self.samples.append(dict(
                track_id=f"{split}_{i:04d}", concept=concept, tags=tags,
                caption_ids=np.array(caption_ids, dtype=np.int64),
                node_feats=feats, edge_index=edge_index,
                valence=valence, arousal=arousal,
                mel=feats.mean(axis=0),  # cheap stand-in "global" audio descriptor for CNN baseline
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def n_tags(self):
        return len(TAG_VOCAB)


def collate_graphs(batch):
    """Batch variable-sized graphs into one big disjoint graph (standard
    GNN batching trick) + a batch-index vector for pooling."""
    node_feats, edge_indices, batch_idx = [], [], []
    offset = 0
    for i, s in enumerate(batch):
        node_feats.append(s["node_feats"])
        ei = s["edge_index"]
        if ei.shape[1] > 0:
            edge_indices.append(ei + offset)
        batch_idx.append(np.full(s["node_feats"].shape[0], i, dtype=np.int64))
        offset += s["node_feats"].shape[0]
    node_feats = torch.tensor(np.concatenate(node_feats, axis=0), dtype=torch.float32)
    edge_index = torch.tensor(np.concatenate(edge_indices, axis=1) if edge_indices
                               else np.zeros((2, 0), dtype=np.int64), dtype=torch.long)
    batch_idx = torch.tensor(np.concatenate(batch_idx), dtype=torch.long)
    tags = torch.tensor(np.stack([s["tags"] for s in batch]), dtype=torch.float32)
    caption_ids = torch.tensor(np.stack([s["caption_ids"] for s in batch]), dtype=torch.long)
    valence = torch.tensor([s["valence"] for s in batch], dtype=torch.float32)
    arousal = torch.tensor([s["arousal"] for s in batch], dtype=torch.float32)
    mel = torch.tensor(np.stack([s["mel"] for s in batch]), dtype=torch.float32)
    return dict(node_feats=node_feats, edge_index=edge_index, batch_idx=batch_idx,
                tags=tags, caption_ids=caption_ids, valence=valence, arousal=arousal,
                mel=mel, n_graphs=len(batch))


def load_tag_names(data_root, dataset_name, n_tags):
    """Reads data/processed/<dataset_name>/tag_names.json if preprocess_dataset.py
    wrote one (e.g. GTZAN genre names); falls back to numeric placeholders
    ('tag_0', 'tag_1', ...) so zero-shot tag prediction (Task 4) always has
    *some* text prompt to embed, even for datasets without a saved vocab."""
    path = os.path.join(data_root, dataset_name, "tag_names.json")
    if os.path.exists(path):
        with open(path) as f:
            names = json.load(f)
        if len(names) == n_tags:
            return names
    return [f"tag_{i}" for i in range(n_tags)]


class ProcessedMusicContextDataset(Dataset):
    """Loads cached graphs + tag/caption/emotion tensors from
    data/processed/<dataset_name>/{split}/*.pt written by
    src/preprocess_dataset.py."""

    def __init__(self, data_root, dataset_name, split):
        self.dir = os.path.join(data_root, dataset_name, split)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.pt")))
        if not self.files:
            raise FileNotFoundError(
                f"No cached samples found under {self.dir}. Run "
                f"src/preprocess_dataset.py first, or use --synthetic for "
                f"a no-download smoke test."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # weights_only=False: these .pt files are written by our own
        # preprocess_dataset.py (trusted, locally generated) and contain
        # plain dicts of tensors/floats/strings, not arbitrary pickled
        # objects — safe to disable PyTorch 2.6+'s stricter default here.
        return torch.load(self.files[idx], weights_only=False)

    @property
    def n_tags(self):
        return self[0]["tags"].shape[0]