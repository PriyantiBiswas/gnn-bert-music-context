"""
make_results.py
----------------
Generates the deliverables required by Section 10 of the project spec:
  * results/metrics.json           (already written by train.py)
  * results/plots/*.png            (loss curves, F1 vs epoch, t-SNE, baseline comparison)
  * data/processed/graphs/*.json   (>=20 example preprocessed graph samples)
  * results/retrieval_examples/*   (Task 4 qualitative retrieval examples)

Run after `train.py --task {1,2,3,4} --synthetic` (or standalone; it will
train fresh lightweight synthetic runs itself).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from datasets import make_synthetic_dataset, train_val_test_split
from bert_encoder import SimpleTextEncoder
from gnn_model import NumpyGraphSAGE
from fusion_model import NumpyFusionModel
from contrastive import NumpyDualEncoder, recall_at_k
from evaluate import (
    macro_micro_f1, auc_pr, mae_r2, baseline_majority, baseline_random, baseline_pca_mlp,
)

PLOTS_DIR = Path("results/plots")
GRAPHS_DIR = Path("data/processed/graphs")
RETRIEVAL_DIR = Path("results/retrieval_examples")
for d in (PLOTS_DIR, GRAPHS_DIR, RETRIEVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)


def save_graph_samples(ds, n: int = 20):
    """Persist >=20 example preprocessed graphs as JSON (nodes/edges/features)."""
    for i, sample in enumerate(ds[:n]):
        g = sample.graph
        payload = {
            "track_id": sample.track_id,
            "genre": sample.genre,
            "text": sample.text,
            "num_nodes": g.num_nodes,
            "node_features": g.x.tolist(),
            "edge_index": [list(e) for e in g.edge_index],
            "edge_weight": list(g.edge_weight),
            "node_names": g.node_names,
        }
        (GRAPHS_DIR / f"{sample.track_id}.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {min(n, len(ds))} graph samples -> {GRAPHS_DIR}")


def plot_loss_curves(histories: dict):
    plt.figure(figsize=(6, 4))
    for name, hist in histories.items():
        plt.plot(hist, label=name)
    plt.xlabel("epoch"); plt.ylabel("loss (log scale)")
    plt.yscale("log")
    plt.title("Training loss curves (Tasks 1-4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curves.png", dpi=150)
    plt.close()


def plot_f1_vs_epoch_task1(texts, targets, epochs: int = 25):
    """Macro/Micro-F1 vs training epoch for the BERT baseline (Task 1 deliverable)."""
    model = SimpleTextEncoder(hidden_dim=32, num_tags=targets.shape[1], seed=SEED)
    macro_hist, micro_hist = [], []
    for _ in range(epochs):
        model.train_step(texts, targets, lr=0.5)
        probs, _ = model.forward(texts)
        preds = (probs > 0.5).astype(np.float32)
        m = macro_micro_f1(targets, preds)
        macro_hist.append(m["macro_f1"]); micro_hist.append(m["micro_f1"])

    plt.figure(figsize=(6, 4))
    plt.plot(macro_hist, label="Macro-F1")
    plt.plot(micro_hist, label="Micro-F1")
    plt.xlabel("epoch"); plt.ylabel("F1")
    plt.title("Task 1: BERT tag classifier — F1 vs epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task1_f1_vs_epoch.png", dpi=150)
    plt.close()
    return macro_hist, micro_hist


def plot_tsne_task3(z_matrix: np.ndarray, genres: list[int], moods: np.ndarray, genre_names):
    """t-SNE of fused representation z, colored by genre and by a mood proxy."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not available; skipping t-SNE plot")
        return

    n = z_matrix.shape[0]
    perplexity = max(2, min(30, n // 3))
    proj = TSNE(n_components=2, random_state=SEED, perplexity=perplexity).fit_transform(z_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc0 = axes[0].scatter(proj[:, 0], proj[:, 1], c=genres, cmap="tab10", s=20)
    axes[0].set_title("t-SNE of fused z, colored by genre")
    handles, _ = sc0.legend_elements()
    axes[0].legend(handles, genre_names, fontsize=6, loc="best")

    sc1 = axes[1].scatter(proj[:, 0], proj[:, 1], c=moods, cmap="coolwarm", s=20)
    axes[1].set_title("t-SNE of fused z, colored by valence")
    plt.colorbar(sc1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task3_tsne.png", dpi=150)
    plt.close()


def plot_baseline_comparison(rows: list[dict]):
    names = [r["model"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(names, f1s)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro-F1")
    plt.title("Baseline comparison (Section 8)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "baseline_comparison.png", dpi=150)
    plt.close()


def main():
    num_tags, num_genres = 12, 8
    ds = make_synthetic_dataset(n_samples=200, num_tags=num_tags, num_genres=num_genres,
                                 graph_type="segment", seed=SEED)
    save_graph_samples(ds, n=20)
    splits = train_val_test_split(ds, seed=SEED)

    # ---- Task 1 F1-vs-epoch + loss curve ----
    train_texts = [s.text for s in splits["train"]]
    train_tags = np.stack([s.tags for s in splits["train"]])
    macro_hist, micro_hist = plot_f1_vs_epoch_task1(train_texts, train_tags, epochs=25)

    histories = {"Task1 (BERT)": [], "Task2 (GNN)": [], "Task3 (Fusion)": [], "Task4 (Contrastive)": []}

    # Task 1 loss curve
    m1 = SimpleTextEncoder(hidden_dim=32, num_tags=num_tags, seed=SEED)
    for _ in range(25):
        histories["Task1 (BERT)"].append(m1.train_step(train_texts, train_tags, lr=0.5))

    # Task 2 loss curve
    in_dim = ds[0].graph.x.shape[1]
    m2 = NumpyGraphSAGE(in_dim=in_dim, hidden_dim=32, out_dim=32, num_layers=2,
                         num_classes=num_genres, seed=SEED)
    genre_targets = np.eye(num_genres, dtype=np.float32)[[s.genre for s in splits["train"]]]
    for _ in range(25):
        histories["Task2 (GNN)"].append(
            m2.train_step([s.graph for s in splits["train"]], genre_targets, lr=0.3)
        )

    # Task 3: fusion loss curve + t-SNE
    text_enc3 = SimpleTextEncoder(hidden_dim=32, num_tags=num_tags, seed=SEED)
    m3 = NumpyFusionModel(g_dim=in_dim, t_dim=32, z_dim=24, num_tags=num_tags, seed=SEED)

    def pairs_for(subset, L=6):
        pairs, tags, emos = [], [], []
        for s in subset:
            g = s.graph.x.mean(axis=0)
            H = np.stack([text_enc3._hash_embed(t) for t in (s.text.split() + [""] * L)[:L]])
            pairs.append((g.astype(np.float32), H.astype(np.float32)))
            tags.append(s.tags); emos.append([s.valence, s.arousal])
        return pairs, np.stack(tags), np.array(emos, dtype=np.float32)

    train_pairs, train_tags3, train_emos3 = pairs_for(splits["train"])
    for _ in range(25):
        histories["Task3 (Fusion)"].append(
            m3.train_step(train_pairs, train_tags3, train_emos3, lr=0.1)
        )

    all_pairs, all_tags3, all_emos3 = pairs_for(ds)
    z_list = [m3.forward(g, H)[2] for g, H in all_pairs]
    plot_tsne_task3(np.stack(z_list), [s.genre for s in ds], all_emos3[:, 0],
                     genre_names=[str(i) for i in range(num_genres)])

    # Task 4: contrastive loss curve + retrieval examples
    text_enc4 = SimpleTextEncoder(hidden_dim=32, num_tags=1, seed=SEED)
    m4 = NumpyDualEncoder(g_dim=in_dim, t_dim=32, embed_dim=64, seed=SEED)

    def featurize4(subset):
        G = np.stack([s.graph.x.mean(axis=0) for s in subset]).astype(np.float32)
        T = text_enc4.encode_text([s.text for s in subset]).astype(np.float32)
        return G, T

    G_train, T_train = featurize4(splits["train"])
    for _ in range(30):
        histories["Task4 (Contrastive)"].append(m4.train_step(G_train, T_train, lr=0.3, tau=0.07))

    G_test, T_test = featurize4(splits["test"])
    g_embed, t_embed = m4.encode(G_test, T_test)
    retrieval_metrics = recall_at_k(g_embed, t_embed, k_values=(1, 5, 10))

    examples = []
    for qi in range(min(10, len(splits["test"]))):
        sim = t_embed[qi] @ g_embed.T
        top3 = np.argsort(-sim)[:3]
        examples.append({
            "query_caption": splits["test"][qi].text,
            "ground_truth_track": splits["test"][qi].track_id,
            "top3_matches": [
                {"track_id": splits["test"][idx].track_id,
                 "text": splits["test"][idx].text,
                 "similarity": float(sim[idx])}
                for idx in top3
            ],
        })
    (RETRIEVAL_DIR / "musiccaps_style_retrieval_examples.json").write_text(
        json.dumps({"retrieval_metrics": retrieval_metrics, "examples": examples}, indent=2)
    )
    print(f"saved 10 qualitative retrieval examples -> {RETRIEVAL_DIR}")

    plot_loss_curves(histories)

    # ---- Baselines (Section 8) ----
    X_train = np.stack([s.graph.x.mean(axis=0) for s in splits["train"]])
    X_test = np.stack([s.graph.x.mean(axis=0) for s in splits["test"]])
    y_train = np.stack([s.tags for s in splits["train"]])
    y_test = np.stack([s.tags for s in splits["test"]])

    b1_maj = baseline_majority(y_train, len(y_test))
    b1_rand = baseline_random(len(y_test), y_test.shape[1], seed=SEED)
    b4_pca_probs = baseline_pca_mlp(X_train, y_train, X_test, n_components=min(16, X_train.shape[1]))
    b4_preds = (b4_pca_probs > 0.5).astype(np.float32)

    task1_probs, _ = m1.forward([s.text for s in splits["test"]])
    task1_preds = (task1_probs > 0.5).astype(np.float32)

    task2_probs = np.stack([m2.forward(s.graph)[0] for s in splits["test"]])
    task2_preds = np.eye(num_genres, dtype=np.float32)[np.argmax(task2_probs, axis=1)]
    # (genre, not tags -- report accuracy separately; skip in tag-F1 bar chart)

    test_pairs3, test_tags3, test_emos3 = pairs_for(splits["test"])
    task3_probs = np.stack([m3.forward(g, H)[0] for g, H in test_pairs3])
    task3_preds = (task3_probs > 0.5).astype(np.float32)

    rows = [
        {"model": "B1: Random", "macro_f1": macro_micro_f1(y_test, b1_rand)["macro_f1"]},
        {"model": "B1: Majority", "macro_f1": macro_micro_f1(y_test, b1_maj)["macro_f1"]},
        {"model": "B4: PCA+MLP", "macro_f1": macro_micro_f1(y_test, b4_preds)["macro_f1"]},
        {"model": "Task1: BERT-only", "macro_f1": macro_micro_f1(y_test, task1_preds)["macro_f1"]},
        {"model": "Task3: GNN-BERT Fusion", "macro_f1": macro_micro_f1(y_test, task3_preds)["macro_f1"]},
    ]
    plot_baseline_comparison(rows)

    results_summary = {
        "task1_f1_vs_epoch": {"macro_f1": macro_hist, "micro_f1": micro_hist},
        "baseline_comparison": rows,
        "task4_retrieval_metrics": retrieval_metrics,
    }
    Path("results/summary.json").write_text(json.dumps(results_summary, indent=2))
    print("wrote results/summary.json")
    print("plots ->", list(PLOTS_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
