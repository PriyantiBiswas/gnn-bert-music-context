"""
train.py
--------
Unified training entrypoint for Tasks 1-4.

    python train.py --task 1 --synthetic --epochs 20
    python train.py --task 2 --synthetic --epochs 20
    python train.py --task 3 --synthetic --epochs 20
    python train.py --task 4 --synthetic --epochs 20

Currently `--synthetic` is the fully wired path (numpy fallback models, so
this runs with zero external dependencies and zero downloads). The
`--dataset {fma_small,fma_medium,magnatagatune,musiccaps,gtzan,deam}` path
calls the corresponding loader in datasets.py — plug in the torch/PyG/
transformers model classes (already implemented in each module, used
automatically when those libraries are importable) once real data is cached
under data/processed/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from datasets import make_synthetic_dataset, train_val_test_split, save_split_ids
from bert_encoder import SimpleTextEncoder
from gnn_model import NumpyGraphSAGE
from fusion_model import NumpyFusionModel
from contrastive import NumpyDualEncoder, recall_at_k
from evaluate import macro_micro_f1, auc_pr, mae_r2


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)


# ---------------------------------------------------------------------------
def run_task1(cfg: dict, epochs: int):
    """BERT-only multi-label tag classifier."""
    num_tags = min(cfg["bert"]["num_tags"], 12)  # keep smoke test small/fast
    ds = make_synthetic_dataset(n_samples=cfg.get("n_samples", 120), num_tags=num_tags,
                                 seed=cfg["seed"])
    splits = train_val_test_split(ds, seed=cfg["seed"])
    save_split_ids(splits, "data/splits/task1_split.json")

    model = SimpleTextEncoder(hidden_dim=128, num_tags=num_tags, seed=cfg["seed"])
    history = []
    for epoch in range(1, epochs + 1):
        texts = [s.text for s in splits["train"]]
        targets = np.stack([s.tags for s in splits["train"]])
        loss = model.train_step(texts, targets, lr=0.5)
        history.append(loss)
        print(f"[Task1][epoch {epoch}/{epochs}] BCE loss={loss:.4f}")

    val_texts = [s.text for s in splits["test"]]
    val_targets = np.stack([s.tags for s in splits["test"]])
    probs, _ = model.forward(val_texts)
    preds = (probs > 0.5).astype(np.float32)
    metrics = macro_micro_f1(val_targets, preds)
    metrics["auc_pr"] = auc_pr(val_targets, probs)
    metrics["loss_curve"] = history
    return metrics


def run_task2(cfg: dict, epochs: int):
    """GNN-only genre classifier on segment graphs."""
    num_genres = cfg["gnn"]["num_genres"]
    ds = make_synthetic_dataset(n_samples=cfg.get("n_samples", 120), num_genres=num_genres,
                                 graph_type="segment", seed=cfg["seed"])
    splits = train_val_test_split(ds, seed=cfg["seed"])
    save_split_ids(splits, "data/splits/task2_split.json")

    in_dim = ds[0].graph.x.shape[1]
    model = NumpyGraphSAGE(in_dim=in_dim, hidden_dim=32, out_dim=32, num_layers=2,
                            num_classes=num_genres, seed=cfg["seed"])
    train_targets = np.eye(num_genres, dtype=np.float32)[[s.genre for s in splits["train"]]]
    history = []
    for epoch in range(1, epochs + 1):
        loss = model.train_step([s.graph for s in splits["train"]], train_targets, lr=0.2)
        history.append(loss)
        print(f"[Task2][epoch {epoch}/{epochs}] BCE loss={loss:.4f}")

    test_targets = np.eye(num_genres, dtype=np.float32)[[s.genre for s in splits["test"]]]
    probs = np.stack([model.forward(s.graph)[0] for s in splits["test"]])
    # single-label genre classification -> argmax (not a 0.5 threshold) for predictions
    pred_idx = np.argmax(probs, axis=1)
    preds = np.eye(num_genres, dtype=np.float32)[pred_idx]
    metrics = macro_micro_f1(test_targets, preds)
    metrics["accuracy"] = float(np.mean(pred_idx == np.argmax(test_targets, axis=1)))
    metrics["loss_curve"] = history
    return metrics


def run_task3(cfg: dict, epochs: int):
    """Cross-attention GNN-BERT fusion + emotion regression."""
    num_tags = min(cfg["bert"]["num_tags"], 12)
    ds = make_synthetic_dataset(n_samples=cfg.get("n_samples", 120), num_tags=num_tags,
                                 graph_type="segment", seed=cfg["seed"])
    splits = train_val_test_split(ds, seed=cfg["seed"])
    save_split_ids(splits, "data/splits/task3_split.json")

    g_dim = ds[0].graph.x.shape[1]
    text_dim = 32
    z_dim = 24
    text_encoder = SimpleTextEncoder(hidden_dim=text_dim, num_tags=num_tags, seed=cfg["seed"])
    model = NumpyFusionModel(g_dim=g_dim, t_dim=text_dim, z_dim=z_dim, num_tags=num_tags,
                              seed=cfg["seed"])

    def sample_pairs(subset):
        pairs, tags, emos = [], [], []
        for s in subset:
            g = s.graph.x.mean(axis=0)  # simple graph-vector for the fallback fusion model
            L = 6
            H_text = np.stack(
                [text_encoder._hash_embed(tok) for tok in (s.text.split() + [""] * L)[:L]]
            )
            pairs.append((g.astype(np.float32), H_text.astype(np.float32)))
            tags.append(s.tags)
            emos.append(np.array([s.valence, s.arousal], dtype=np.float32))
        return pairs, np.stack(tags), np.stack(emos)

    train_pairs, train_tags, train_emos = sample_pairs(splits["train"])
    history = []
    for epoch in range(1, epochs + 1):
        loss = model.train_step(train_pairs, train_tags, train_emos, lr=0.1,
                                 alpha=cfg["fusion"]["emotion_alpha"],
                                 beta=cfg["fusion"]["emotion_beta"])
        history.append(loss)
        print(f"[Task3][epoch {epoch}/{epochs}] multi-task loss={loss:.4f}")

    test_pairs, test_tags, test_emos = sample_pairs(splits["test"])
    tag_preds, emo_preds = [], []
    for g, H_text in test_pairs:
        probs, emo, _, _ = model.forward(g, H_text)
        tag_preds.append(probs)
        emo_preds.append(emo)
    tag_preds = np.stack(tag_preds)
    emo_preds = np.stack(emo_preds)

    metrics = macro_micro_f1(test_tags, (tag_preds > 0.5).astype(np.float32))
    metrics["auc_pr"] = auc_pr(test_tags, tag_preds)
    metrics["valence"] = mae_r2(test_emos[:, 0], emo_preds[:, 0])
    metrics["arousal"] = mae_r2(test_emos[:, 1], emo_preds[:, 1])
    metrics["loss_curve"] = history
    return metrics


def run_task4(cfg: dict, epochs: int):
    """Contrastive InfoNCE dual-encoder (MusicCaps-style graph<->caption retrieval)."""
    ds = make_synthetic_dataset(n_samples=cfg.get("n_samples", 80), graph_type="segment",
                                 seed=cfg["seed"])
    splits = train_val_test_split(ds, seed=cfg["seed"])
    save_split_ids(splits, "data/splits/task4_split.json")

    g_dim = ds[0].graph.x.shape[1]
    t_dim = 32
    text_encoder = SimpleTextEncoder(hidden_dim=t_dim, num_tags=1, seed=cfg["seed"])
    model = NumpyDualEncoder(g_dim=g_dim, t_dim=t_dim,
                              embed_dim=cfg["contrastive"]["embed_dim"] // 4,
                              seed=cfg["seed"])

    def featurize(subset):
        G = np.stack([s.graph.x.mean(axis=0) for s in subset]).astype(np.float32)
        T = text_encoder.encode_text([s.text for s in subset]).astype(np.float32)
        return G, T

    G_train, T_train = featurize(splits["train"])
    history = []
    for epoch in range(1, epochs + 1):
        loss = model.train_step(G_train, T_train, lr=0.3,
                                 tau=cfg["contrastive"]["temperature"])
        history.append(loss)
        print(f"[Task4][epoch {epoch}/{epochs}] InfoNCE loss={loss:.4f}")

    G_test, T_test = featurize(splits["test"])
    g_embed, t_embed = model.encode(G_test, T_test)
    metrics = recall_at_k(g_embed, t_embed, k_values=tuple(cfg["eval"]["k_values"]))
    metrics["loss_curve"] = history
    return metrics


TASKS = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}


def main():
    parser = argparse.ArgumentParser(description="Train GNN-BERT music context models.")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--synthetic", action="store_true",
                         help="Use synthetic data (default/only supported path currently).")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Real dataset name (fma_small|fma_medium|magnatagatune|"
                              "musiccaps|gtzan|deam) — requires data under data/raw/.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, default="results/metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    epochs = args.epochs or cfg["train"]["epochs"]

    if args.dataset and not args.synthetic:
        raise NotImplementedError(
            "Real-dataset training requires torch/torch_geometric/transformers "
            "and preprocessed data under data/processed/. See datasets.py loaders "
            "and swap the Numpy* fallback classes in train.py for the torch model "
            "classes (BertTagClassifier / GNNClassifier / GNNBertFusionModel / "
            "DualEncoder), which are used automatically when those libraries "
            "are importable."
        )

    t0 = time.time()
    metrics = TASKS[args.task](cfg, epochs)
    metrics["elapsed_sec"] = round(time.time() - t0, 2)

    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if Path(args.out).exists():
        existing = json.loads(Path(args.out).read_text())
    existing[f"task{args.task}"] = _to_jsonable(metrics)
    Path(args.out).write_text(json.dumps(existing, indent=2))

    print(f"\n=== Task {args.task} final metrics ===")
    for k, v in metrics.items():
        if k != "loss_curve":
            print(f"  {k}: {v}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
