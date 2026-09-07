# gnn-bert-music-context

**GNN-Based BERT for Understanding Context from Music**
Course: Neural Networks (CSE425 / EEE474 / CSE715) — Supervised Neural Network Project
Prepared by: Moin Mostakim · Deadline: 2 Oct 2026

A hybrid **BERT + Graph Neural Network** system for understanding musical context:
multi-label tagging, mood/emotion regression, and cross-modal audio↔text alignment.
BERT encodes lyrics / tags / captions; a GNN (GraphSAGE) message-passes over music
structure graphs (chord-transition graphs and segment-similarity graphs); a
cross-attention fusion head combines both for downstream prediction.

## Project structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            # FMA, MagnaTagATune, MusicCaps downloads (not tracked)
│   ├── processed/      # cached graphs (.pt), mel/chroma features, BERT caches
│   └── splits/         # train/val/test JSON, artist-disjoint
├── notebooks/
│   ├── eda.ipynb              # dataset exploration
│   └── demo_context.ipynb     # end-to-end inference demo (synthetic-data runnable)
├── src/
│   ├── audio_features.py      # mel / chroma extraction, segmentation (librosa)
│   ├── graph_builder.py       # chord-transition & segment-similarity graphs
│   ├── bert_encoder.py        # Task 1: BERT multi-label tag classifier
│   ├── gnn_model.py           # Task 2: GraphSAGE / GAT encoder
│   ├── fusion_model.py        # Task 3: cross-attention GNN–BERT fusion
│   ├── contrastive.py         # Task 4: InfoNCE dual-encoder + retrieval
│   ├── datasets.py            # dataset loaders (FMA/MagnaTagATune/MusicCaps/DEAM) + synthetic fallback
│   ├── train.py               # unified training entrypoint (--task 1|2|3|4)
│   └── evaluate.py            # Macro/Micro-F1, AUC-PR, MAE/R², R@K
├── results/
│   ├── metrics.json
│   ├── plots/
│   └── retrieval_examples/
└── report/
    └── final_report.pdf       # (student-authored, NeurIPS/IEEE/ICML template)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (synthetic smoke test — no dataset download needed)

Every module has a `--synthetic` fallback so the whole pipeline can be run and
validated end-to-end before wiring up real datasets:

```bash
# Task 1: BERT tag classifier
python src/train.py --task 1 --synthetic --epochs 3

# Task 2: GNN on segment/chord graphs
python src/train.py --task 2 --synthetic --epochs 3

# Task 3: GNN–BERT cross-attention fusion (+ optional emotion regression)
python src/train.py --task 3 --synthetic --epochs 3

# Task 4: Contrastive GNN–BERT (MusicCaps-style retrieval)
python src/train.py --task 4 --synthetic --epochs 3

# End-to-end inference demo
jupyter notebook notebooks/demo_context.ipynb
```

Swap `--synthetic` for `--data-root data/processed --dataset fma_small` (etc.)
once real preprocessed graphs/features are cached under `data/processed/`
(see `src/datasets.py` and Section 3 of the project spec for the preprocessing
pipeline: resample → mel/chroma → segment/chord graph → BERT tokenize → split).

## Tasks implemented

| Task | Model | File | Metric |
|---|---|---|---|
| 1 — Easy | BERT multi-label tag classifier | `bert_encoder.py` | Macro/Micro-F1 |
| 2 — Medium | GraphSAGE on segment/chord graph | `gnn_model.py` | Macro-F1 vs CNN baseline |
| 3 — Hard | Cross-attention GNN–BERT fusion + emotion regression | `fusion_model.py` | Macro-F1, AUC-PR, MAE |
| 4 — Advanced | InfoNCE contrastive dual-encoder | `contrastive.py` | R@1/5/10 |

## Baselines (`src/evaluate.py --baselines`)
- B1: majority-class / random tag predictor
- B2: CNN on mel-spectrogram (no graph, no text)
- B3: BERT-only (Task 1 model)
- B4 (optional): PCA + MLP on hand-crafted audio features

## Reproducibility notes
- All random seeds are set via `config.yaml: seed`.
- Splits are artist-disjoint where artist metadata is available (FMA/MSD).
- `data/processed/` graph cache uses PyTorch Geometric `Data` objects saved as `.pt`.
.


NEW