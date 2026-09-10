"""
Unified training entrypoint.

    python src/train.py --task 1 --synthetic --epochs 3
    python src/train.py --task 2 --synthetic --epochs 3
    python src/train.py --task 3 --synthetic --epochs 3 --fusion cross_attention
    python src/train.py --task 4 --synthetic --epochs 3

    python src/train.py --task 1 --data-root data/processed --dataset gtzan --epochs 15
    python src/train.py --task 2 --data-root data/processed --dataset gtzan --epochs 30
    python src/train.py --task 3 --data-root data/processed --dataset gtzan --ablation --epochs 30
    python src/train.py --task 4 --data-root data/processed --dataset gtzan --epochs 30

Writes results/metrics.json (merged across runs), results/plots/*.png,
and (Task 4) results/retrieval_examples/*.json.
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_seed, load_config, save_json, device
from src.datasets import (
    SyntheticMusicContextDataset, ProcessedMusicContextDataset,
    collate_graphs, VOCAB_SIZE, load_tag_names, TAG_VOCAB,
)
from src.bert_encoder import MiniTextEncoder, BertTagClassifier, bce_loss, compute_pos_weight
from src.gnn_model import GraphSAGE, GNNTagClassifier, CNNBaseline
from src.fusion_model import GNNBertFusion, multitask_loss
from src.contrastive import DualEncoder, info_nce_loss, retrieval_recall_at_k
from src.evaluate import tag_metrics, emotion_metrics, random_baseline, pca_mlp_baseline
from src.graph_builder import graph_coherence_score

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS = os.path.join(RESULTS, "plots")
RETRIEVAL = os.path.join(RESULTS, "retrieval_examples")
METRICS_PATH = os.path.join(RESULTS, "metrics.json")


def _update_metrics(key, value):
    os.makedirs(RESULTS, exist_ok=True)
    all_metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            all_metrics = json.load(f)
    all_metrics[key] = value
    save_json(all_metrics, METRICS_PATH)


def _plot_curve(history: dict, title: str, fname: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS, exist_ok=True)
    plt.figure(figsize=(5, 4))
    for k, v in history.items():
        plt.plot(range(1, len(v) + 1), v, marker="o", label=k)
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close()


def _samples(ds):
    """Uniform accessor: SyntheticMusicContextDataset has .samples (list of
    dicts); ProcessedMusicContextDataset lazy-loads via __getitem__. Returns
    a list either way (materializes real data once, fine at this scale)."""
    return ds.samples if hasattr(ds, "samples") else [ds[i] for i in range(len(ds))]


def get_loaders(args, cfg):
    if args.synthetic:
        train_ds = SyntheticMusicContextDataset(n_samples=args.n_samples, split="train",
                                                 seed=cfg["seed"], max_length=cfg["text"]["max_length"])
        val_ds = SyntheticMusicContextDataset(n_samples=max(64, args.n_samples // 4), split="val",
                                               seed=cfg["seed"], max_length=cfg["text"]["max_length"])
        test_ds = SyntheticMusicContextDataset(n_samples=max(64, args.n_samples // 4), split="test",
                                                seed=cfg["seed"], max_length=cfg["text"]["max_length"])
    else:
        train_ds = ProcessedMusicContextDataset(args.data_root, args.dataset, "train")
        val_ds = ProcessedMusicContextDataset(args.data_root, args.dataset, "val")
        test_ds = ProcessedMusicContextDataset(args.data_root, args.dataset, "test")

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                                         shuffle=shuffle, collate_fn=collate_graphs)
    return train_ds, val_ds, test_ds, mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def _tag_names_for(args, n_tags):
    if args.synthetic:
        return TAG_VOCAB if len(TAG_VOCAB) == n_tags else [f"tag_{i}" for i in range(n_tags)]
    return load_tag_names(args.data_root, args.dataset, n_tags)


# --------------------------------------------------------------------------- #
# Task 1: BERT tag classifier
# --------------------------------------------------------------------------- #
def run_task1(args, cfg, dev):
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = get_loaders(args, cfg)
    encoder = MiniTextEncoder(VOCAB_SIZE, cfg["text"]["embed_dim"], cfg["text"]["n_layers"],
                               cfg["text"]["n_heads"], cfg["text"]["max_length"]).to(dev)
    model = BertTagClassifier(encoder, train_ds.n_tags).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    pos_weight = compute_pos_weight(np.stack([s["tags"] for s in _samples(train_ds)]), dev)

    hist = {"train_macro_f1": [], "val_macro_f1": []}
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dl:
            opt.zero_grad()
            logits = model(batch["caption_ids"].to(dev))
            loss = bce_loss(logits, batch["tags"].to(dev), pos_weight)
            loss.backward()
            opt.step()
        tr_m = _eval_task1(model, train_dl, dev)
        va_m = _eval_task1(model, val_dl, dev)
        hist["train_macro_f1"].append(tr_m["macro_f1"])
        hist["val_macro_f1"].append(va_m["macro_f1"])
        print(f"[Task1][epoch {epoch+1}/{args.epochs}] loss={loss.item():.4f} "
              f"train_macroF1={tr_m['macro_f1']:.3f} val_macroF1={va_m['macro_f1']:.3f}")

    test_m = _eval_task1(model, test_dl, dev)
    print("[Task1] TEST:", test_m)
    _plot_curve(hist, "Task 1: BERT tag classifier — Macro-F1", "task1_f1_curve.png")

    tag_names = _tag_names_for(args, train_ds.n_tags)
    model.eval()
    examples = []
    with torch.no_grad():
        batch = next(iter(test_dl))
        probs = torch.sigmoid(model(batch["caption_ids"].to(dev))).cpu().numpy()
        for i in range(min(5, probs.shape[0])):
            true_tags = [tag_names[k] for k in range(len(tag_names)) if batch["tags"][i, k] == 1]
            pred_tags = [tag_names[k] for k in np.argsort(-probs[i])[:4]]
            examples.append({"true_tags": true_tags, "predicted_top4": pred_tags})
    save_json(examples, os.path.join(RESULTS, "task1_example_predictions.json"))

    y_true_test = np.stack([s["tags"] for s in _samples(test_ds)])
    baseline = random_baseline(y_true_test)
    _update_metrics("task1_bert", {"test": test_m, "history": hist})
    _update_metrics("baseline_B1_random", baseline)
    return test_m


def _eval_task1(model, dl, dev):
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["caption_ids"].to(dev))
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(batch["tags"].numpy())
    return tag_metrics(np.concatenate(all_true), np.concatenate(all_prob))


# --------------------------------------------------------------------------- #
# Task 2: GNN on segment/chord graphs, vs CNN baseline, vs B4 (PCA+MLP)
# --------------------------------------------------------------------------- #
def run_task2(args, cfg, dev):
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = get_loaders(args, cfg)
    train_samples = _samples(train_ds)
    test_samples = _samples(test_ds)
    in_dim = train_samples[0]["node_feats"].shape[1]
    mel_dim = train_samples[0]["mel"].shape[0]

    model = GNNTagClassifier(in_dim, train_ds.n_tags, cfg["gnn"]["hidden_dim"],
                              cfg["gnn"]["n_layers"], cfg["gnn"]["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    cnn = CNNBaseline(mel_dim, train_ds.n_tags, cfg["gnn"]["hidden_dim"]).to(dev)
    cnn_opt = torch.optim.Adam(cnn.parameters(), lr=cfg["training"]["lr"])
    pos_weight = compute_pos_weight(np.stack([s["tags"] for s in train_samples]), dev)

    hist = {"gnn_val_macro_f1": [], "cnn_val_macro_f1": []}
    for epoch in range(args.epochs):
        model.train(); cnn.train()
        for batch in train_dl:
            opt.zero_grad()
            logits, _ = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                               batch["batch_idx"].to(dev), batch["n_graphs"])
            loss = bce_loss(logits, batch["tags"].to(dev), pos_weight)
            loss.backward(); opt.step()

            cnn_opt.zero_grad()
            cnn_logits = cnn(batch["mel"].to(dev))
            cnn_loss = bce_loss(cnn_logits, batch["tags"].to(dev), pos_weight)
            cnn_loss.backward(); cnn_opt.step()

        gnn_val = _eval_task2_gnn(model, val_dl, dev)
        cnn_val = _eval_task2_cnn(cnn, val_dl, dev)
        hist["gnn_val_macro_f1"].append(gnn_val["macro_f1"])
        hist["cnn_val_macro_f1"].append(cnn_val["macro_f1"])
        print(f"[Task2][epoch {epoch+1}/{args.epochs}] "
              f"GNN val_macroF1={gnn_val['macro_f1']:.3f}  CNN(B2) val_macroF1={cnn_val['macro_f1']:.3f}")

    gnn_test = _eval_task2_gnn(model, test_dl, dev)
    cnn_test = _eval_task2_cnn(cnn, test_dl, dev)
    print("[Task2] TEST GNN:", gnn_test, " CNN(B2):", cnn_test)
    _plot_curve(hist, "Task 2: GNN vs CNN baseline — Macro-F1", "task2_gnn_vs_cnn.png")
    _update_metrics("task2_gnn", {"test": gnn_test, "history": hist})
    _update_metrics("baseline_B2_cnn", cnn_test)

    # --- B4: PCA + MLP on hand-crafted (pooled mel) audio features ---
    X_train = np.stack([s["mel"].numpy() if torch.is_tensor(s["mel"]) else s["mel"] for s in train_samples])
    y_train = np.stack([s["tags"].numpy() if torch.is_tensor(s["tags"]) else s["tags"] for s in train_samples])
    X_test = np.stack([s["mel"].numpy() if torch.is_tensor(s["mel"]) else s["mel"] for s in test_samples])
    y_test = np.stack([s["tags"].numpy() if torch.is_tensor(s["tags"]) else s["tags"] for s in test_samples])
    b4_result = pca_mlp_baseline(X_train, y_train, X_test, y_test, n_components=min(16, X_train.shape[1]),
                                  seed=cfg["seed"])
    print("[Task2] TEST B4 (PCA+MLP):", b4_result)
    _update_metrics("baseline_B4_pca_mlp", b4_result)

    # --- Graph coherence score (Section 6, optional analysis): do the GNN's
    # learned node embeddings agree with the graph's own similarity edges? ---
    coherence = _graph_coherence_over_dataset(model, test_samples, dev)
    print(f"[Task2] TEST graph coherence score: {coherence:.3f}")
    _update_metrics("task2_graph_coherence", {"mean_coherence": coherence})

    return gnn_test


def _eval_task2_gnn(model, dl, dev):
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch in dl:
            logits, _ = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                               batch["batch_idx"].to(dev), batch["n_graphs"])
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(batch["tags"].numpy())
    return tag_metrics(np.concatenate(all_true), np.concatenate(all_prob))


def _eval_task2_cnn(model, dl, dev):
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["mel"].to(dev))
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(batch["tags"].numpy())
    return tag_metrics(np.concatenate(all_true), np.concatenate(all_prob))


def _graph_coherence_over_dataset(model, samples, dev, max_graphs=50, tau=0.6):
    """Runs the trained GNN's message-passing layers (not the tag head) on
    each graph individually (unbatched, so edge_index stays in local node
    indices) and averages graph_builder.graph_coherence_score across up to
    max_graphs test graphs."""
    model.eval()
    scores = []
    with torch.no_grad():
        for s in samples[:max_graphs]:
            nf = s["node_feats"]
            nf = torch.as_tensor(nf, dtype=torch.float32).to(dev)
            ei = s["edge_index"]
            ei = torch.as_tensor(ei, dtype=torch.long).to(dev)
            if ei.numel() == 0 or nf.shape[0] < 2:
                continue
            h = model.gnn(nf, ei)  # node embeddings, pre-readout
            score = graph_coherence_score(h.cpu().numpy(), ei.cpu().numpy(), tau=tau)
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0



# --------------------------------------------------------------------------- #
# Task 3: GNN-BERT fusion (+ ablations) and emotion regression
# --------------------------------------------------------------------------- #
def run_task3(args, cfg, dev):
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = get_loaders(args, cfg)
    train_samples = _samples(train_ds)
    in_dim = train_samples[0]["node_feats"].shape[1]
    all_tags = np.stack([s["tags"] for s in train_samples])

    modes = ["bert_only", "gnn_only", "concat", "cross_attention"] if args.ablation else [args.fusion]
    ablation_results = {}
    final_model, final_mode = None, None
    for mode in modes:
        gnn = GraphSAGE(in_dim, cfg["gnn"]["hidden_dim"], cfg["gnn"]["n_layers"], cfg["gnn"]["dropout"])
        text = MiniTextEncoder(VOCAB_SIZE, cfg["text"]["embed_dim"], cfg["text"]["n_layers"],
                                cfg["text"]["n_heads"], cfg["text"]["max_length"])
        model = GNNBertFusion(gnn, text, train_ds.n_tags, fusion_mode=mode).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"] * 0.5)
        pos_weight = compute_pos_weight(all_tags, dev)

        hist = {"val_macro_f1": [], "val_mae": []}
        for epoch in range(args.epochs):
            model.train()
            for batch in train_dl:
                opt.zero_grad()
                tag_logits, va_pred, _ = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                                                batch["batch_idx"].to(dev), batch["n_graphs"],
                                                batch["caption_ids"].to(dev))
                va_true = torch.stack([batch["valence"], batch["arousal"]], dim=1).to(dev)
                loss, _ = multitask_loss(tag_logits, batch["tags"].to(dev), va_pred, va_true,
                                          pos_weight=pos_weight)
                loss.backward(); opt.step()
            val_m = _eval_task3(model, val_dl, dev)
            hist["val_macro_f1"].append(val_m["macro_f1"])
            hist["val_mae"].append(val_m["mae"])
            print(f"[Task3:{mode}][epoch {epoch+1}/{args.epochs}] "
                  f"val_macroF1={val_m['macro_f1']:.3f} val_MAE={val_m['mae']:.3f}")

        test_m = _eval_task3(model, test_dl, dev)
        print(f"[Task3:{mode}] TEST:", test_m)
        ablation_results[mode] = {"test": test_m, "history": hist}
        if mode == (args.fusion if not args.ablation else "cross_attention"):
            final_model, final_mode = model, mode

    _plot_curve({m: ablation_results[m]["history"]["val_macro_f1"] for m in ablation_results},
                "Task 3: fusion ablation — val Macro-F1", "task3_ablation_f1.png")
    _update_metrics("task3_fusion_ablation", ablation_results)

    if final_model is not None:
        _tsne_plot(final_model, test_dl, dev, _samples(test_ds))
        _case_studies(final_model, test_dl, dev, _tag_names_for(args, train_ds.n_tags))
    return ablation_results


def _eval_task3(model, dl, dev):
    model.eval()
    all_true, all_prob, va_true_l, va_pred_l = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            tag_logits, va_pred, _ = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                                            batch["batch_idx"].to(dev), batch["n_graphs"],
                                            batch["caption_ids"].to(dev))
            all_prob.append(torch.sigmoid(tag_logits).cpu().numpy())
            all_true.append(batch["tags"].numpy())
            va_true_l.append(torch.stack([batch["valence"], batch["arousal"]], dim=1).numpy())
            va_pred_l.append(va_pred.cpu().numpy())
    tm = tag_metrics(np.concatenate(all_true), np.concatenate(all_prob))
    em = emotion_metrics(np.concatenate(va_true_l), np.concatenate(va_pred_l))
    return {**tm, **em}


def _tsne_plot(model, dl, dev, test_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model.eval()
    zs = []
    with torch.no_grad():
        for batch in dl:
            from src.gnn_model import GraphSAGE
            h = model.gnn(batch["node_feats"].to(dev), batch["edge_index"].to(dev))
            g = GraphSAGE.readout(h, batch["batch_idx"].to(dev), batch["n_graphs"])
            cls, h_text = model.text(batch["caption_ids"].to(dev))
            if model.fusion_mode == "cross_attention":
                z, _ = model.fusion(g, h_text)
            else:
                z = torch.cat([g, cls], dim=-1)
            zs.append(z.cpu().numpy())
    concepts = np.array([int(s.get("concept", 0)) for s in test_samples])
    Z = np.concatenate(zs)
    n = min(len(Z), len(concepts))
    Z, concepts = Z[:n], concepts[:n]
    if n < 4:
        print(f"[Task3] skipping t-SNE plot — only {n} test samples, too few for a meaningful embedding.")
        return
    perplex = max(2, min(30, n // 3, n - 1))
    Z2 = TSNE(n_components=2, perplexity=perplex, random_state=42, init="pca").fit_transform(Z)
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=concepts, cmap="tab10", s=18)
    plt.legend(*sc.legend_elements(), title="concept/genre", loc="best", fontsize=7)
    plt.title("Task 3: t-SNE of fused embeddings z")
    plt.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    plt.savefig(os.path.join(PLOTS, "task3_tsne.png"), dpi=120)
    plt.close()


def _case_studies(model, dl, dev, tag_names, n=3):
    model.eval()
    batch = next(iter(dl))
    with torch.no_grad():
        tag_logits, va_pred, attn = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                                           batch["batch_idx"].to(dev), batch["n_graphs"],
                                           batch["caption_ids"].to(dev))
    probs = torch.sigmoid(tag_logits).cpu().numpy()
    cases = []
    for i in range(min(n, probs.shape[0])):
        entry = {
            "predicted_top_tags": [tag_names[k] for k in np.argsort(-probs[i])[:3]],
            "predicted_valence_arousal": va_pred[i].tolist(),
        }
        if attn is not None:
            entry["graph_to_caption_attention_over_tokens"] = attn[i].cpu().numpy().round(3).tolist()
        cases.append(entry)
    save_json(cases, os.path.join(RESULTS, "task3_case_studies.json"))


# --------------------------------------------------------------------------- #
# Task 4: contrastive dual-encoder + retrieval + zero-shot tag prediction
# --------------------------------------------------------------------------- #
def run_task4(args, cfg, dev):
    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = get_loaders(args, cfg)
    train_samples = _samples(train_ds)
    in_dim = train_samples[0]["node_feats"].shape[1]
    gnn = GraphSAGE(in_dim, cfg["gnn"]["hidden_dim"], cfg["gnn"]["n_layers"], cfg["gnn"]["dropout"])
    text = MiniTextEncoder(VOCAB_SIZE, cfg["text"]["embed_dim"], cfg["text"]["n_layers"],
                            cfg["text"]["n_heads"], cfg["text"]["max_length"])
    model = DualEncoder(gnn, text, cfg["contrastive"]["embed_dim"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    temp = cfg["contrastive"]["temperature"]

    hist = {"train_loss": [], "val_R@5_a2c": []}
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_dl:
            opt.zero_grad()
            g_e, t_e = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                              batch["batch_idx"].to(dev), batch["n_graphs"], batch["caption_ids"].to(dev))
            loss = info_nce_loss(g_e, t_e, temp)
            loss.backward(); opt.step()
            epoch_loss += loss.item()
        val_r = _eval_task4(model, val_dl, dev)
        hist["train_loss"].append(epoch_loss / max(1, len(train_dl)))
        hist["val_R@5_a2c"].append(val_r.get("audio_to_caption_R@5", 0.0))
        print(f"[Task4][epoch {epoch+1}/{args.epochs}] loss={hist['train_loss'][-1]:.4f} "
              f"val R@5(audio->caption)={hist['val_R@5_a2c'][-1]:.3f}")

    test_r = _eval_task4(model, test_dl, dev, save_examples=True, test_samples=_samples(test_ds))
    print("[Task4] TEST retrieval:", test_r)
    _plot_curve({"train_loss": hist["train_loss"]}, "Task 4: contrastive InfoNCE loss", "task4_loss_curve.png")
    _plot_curve({"val_R@5_a2c": hist["val_R@5_a2c"]}, "Task 4: val Audio->Caption R@5", "task4_recall_curve.png")
    _update_metrics("task4_contrastive", {"test": test_r, "history": hist})

    # --- Zero-shot tag prediction from captions, vs. Task 3 supervised ---
    # (spec Section 4.4 deliverable: "Zero-shot tag prediction from captions
    # vs. Task 3 supervised model")
    tag_names = _tag_names_for(args, train_ds.n_tags)
    zero_shot = _zero_shot_tag_eval(model, test_dl, tag_names, cfg["text"]["max_length"], dev)
    print("[Task4] TEST zero-shot tag prediction:", zero_shot)
    _update_metrics("task4_zero_shot_tag_prediction", zero_shot)
    _compare_zero_shot_vs_task3(zero_shot)

    return test_r


def _eval_task4(model, dl, dev, save_examples=False, test_samples=None):
    model.eval()
    g_all, t_all = [], []
    with torch.no_grad():
        for batch in dl:
            g_e, t_e = model(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                              batch["batch_idx"].to(dev), batch["n_graphs"], batch["caption_ids"].to(dev))
            g_all.append(g_e.cpu()); t_all.append(t_e.cpu())
    g_all, t_all = torch.cat(g_all), torch.cat(t_all)
    metrics = retrieval_recall_at_k(g_all, t_all)
    if save_examples and test_samples is not None:
        sims = (g_all @ t_all.t()).numpy()
        examples = []
        for i in range(min(10, len(test_samples))):
            top3 = np.argsort(-sims[i])[:3].tolist()
            examples.append({
                "query_track": test_samples[i].get("track_id", str(i)),
                "top3_retrieved_track_ids": [test_samples[j].get("track_id", str(j)) for j in top3],
                "correct_in_top3": bool(i in top3),
            })
        save_json(examples, os.path.join(RETRIEVAL, "task4_retrieval_examples.json"))
    return metrics


def _zero_shot_tag_eval(model, test_dl, tag_names, max_length, dev):
    """Embeds each tag name as a short text prompt ("a song about <tag>")
    with the Task 4 dual encoder's *text tower only* (no classifier head was
    ever trained for this), then scores each test clip by cosine similarity
    between its graph embedding and every tag-prompt embedding. This is
    genuinely zero-shot: the tag head from Task 1/3 is never used here."""
    from src.datasets import tokenize
    model.eval()
    prompts = [tokenize(["a", "song", "about", name], max_length) for name in tag_names]
    prompt_ids = torch.tensor(prompts, dtype=torch.long, device=dev)
    with torch.no_grad():
        tag_embeds = model.encode_text(prompt_ids)  # [K, D], L2-normalized

    all_true, all_sim = [], []
    with torch.no_grad():
        for batch in test_dl:
            g_e = model.encode_graph(batch["node_feats"].to(dev), batch["edge_index"].to(dev),
                                      batch["batch_idx"].to(dev), batch["n_graphs"])
            sims = g_e @ tag_embeds.t()  # cosine similarity, both sides normalized, in [-1, 1]
            all_sim.append(sims.cpu().numpy())
            all_true.append(batch["tags"].numpy())
    y_true = np.concatenate(all_true)
    # squash cosine similarity into a pseudo-probability for F1/AUC-PR scoring
    y_prob = 1.0 / (1.0 + np.exp(-5.0 * np.concatenate(all_sim)))
    return tag_metrics(y_true, y_prob)


def _compare_zero_shot_vs_task3(zero_shot_metrics):
    """Reads task3's already-saved metrics (if present) and logs a direct
    zero-shot-vs-supervised comparison, per the spec's explicit deliverable."""
    if not os.path.exists(METRICS_PATH):
        return
    with open(METRICS_PATH) as f:
        all_metrics = json.load(f)
    task3 = all_metrics.get("task3_fusion_ablation")
    if not task3:
        print("[Task4] (Task 3 not yet run in this results/metrics.json — "
              "run Task 3 first for a zero-shot-vs-supervised comparison.)")
        return
    # prefer the cross_attention entry if the ablation was run, else whatever's there
    task3_entry = task3.get("cross_attention", next(iter(task3.values())))
    comparison = {
        "zero_shot_task4_macro_f1": zero_shot_metrics["macro_f1"],
        "zero_shot_task4_mean_aucpr": zero_shot_metrics["mean_aucpr"],
        "supervised_task3_macro_f1": task3_entry["test"]["macro_f1"],
        "supervised_task3_mean_aucpr": task3_entry["test"]["mean_aucpr"],
    }
    print("[Task4] zero-shot vs. Task 3 supervised:", comparison)
    _update_metrics("zero_shot_vs_task3_supervised", comparison)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--synthetic", action="store_true", help="use synthetic in-memory dataset")
    p.add_argument("--data-root", type=str, default="data/processed")
    p.add_argument("--dataset", type=str, default="gtzan")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=480)
    p.add_argument("--fusion", type=str, default="cross_attention",
                    choices=["bert_only", "gnn_only", "concat", "cross_attention"])
    p.add_argument("--ablation", action="store_true", help="Task 3: run all 4 fusion modes")
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs is None:
        args.epochs = cfg["training"]["epochs"]
    set_seed(cfg["seed"])
    dev = device()
    print(f"[setup] device={dev}  task={args.task}  synthetic={args.synthetic}  epochs={args.epochs}")

    if args.task == 1:
        run_task1(args, cfg, dev)
    elif args.task == 2:
        run_task2(args, cfg, dev)
    elif args.task == 3:
        run_task3(args, cfg, dev)
    elif args.task == 4:
        run_task4(args, cfg, dev)


if __name__ == "__main__":
    main()