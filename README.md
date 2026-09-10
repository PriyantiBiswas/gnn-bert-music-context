# gnn-bert-music-context

**GNN-Based BERT for Understanding Context from Music**
Course: Neural Networks (CSE425 / EEE474 / CSE715) — Supervised Neural Network Project
Prepared by: [your name] · Deadline: 2 Oct 2026

A hybrid **BERT + Graph Neural Network** system for understanding musical context:
multi-label tagging, mood/emotion regression, and cross-modal audio↔text alignment.
A text encoder reads captions/tags; a GNN (GraphSAGE) message-passes over
segment-similarity graphs built from audio features; a cross-attention fusion
head combines both for downstream prediction; a contrastive dual-encoder
aligns audio and text into a shared embedding space for retrieval.

## Project structure
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│ ├── raw/
│ │ ├── gtzan/<10 genre folders>/.wav # GTZAN download
│ │ └── deam/MEMD_audio/.mp3, annotations/ # DEAM download
│ ├── processed/
│ │ ├── gtzan/{train,val,test}/.pt, tag_names.json
│ │ └── deam/{train,val,test}/.pt, tag_names.json
│ └── splits/
│ ├── gtzan_splits.json # train/val/test track_id manifest
│ └── deam_splits.json # + real valence/arousal/quadrant per track
├── notebooks/
│ ├── eda.ipynb
│ └── demo_context.ipynb
├── src/
│ ├── audio_features.py # mel-spectrogram extraction, segmentation (librosa)
│ ├── graph_builder.py # segment-similarity & chord-transition graphs
│ ├── preprocess_dataset.py # GTZAN: audio -> graphs, genre tags, descriptive captions
│ ├── preprocess_deam.py # DEAM: audio -> graphs, mood-quadrant tags/captions from real valence/arousal
│ ├── bert_encoder.py # Task 1: text encoder (MiniTextEncoder) + tag classifier
│ ├── gnn_model.py # Task 2: GraphSAGE / GAT encoder + CNN baseline (B2)
│ ├── fusion_model.py # Task 3: cross-attention GNN-BERT fusion + emotion regression head
│ ├── contrastive.py # Task 4: InfoNCE dual-encoder + retrieval R@K
│ ├── datasets.py # synthetic corpus (offline smoke test) + real .pt loader + vocabulary
│ ├── evaluate.py # Macro/Micro-F1, AUC-PR, MAE/R², baselines B1 & B4
│ └── train.py # unified entrypoint (--task 1|2|3|4, --synthetic or --dataset ...)
├── results/
│ ├── metrics.json # all tasks + baselines, merged by key (back up before switching datasets!)
│ ├── plots/ # *.png per task
│ └── retrieval_examples/ # Task 4 qualitative examples
└── report/
└── final_report.pdf



## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows; use `source venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```


## Real datasets used

| Dataset | Used for | Why |
|---|---|---|
| **GTZAN** | Tasks 1, 2, 3, 4 | 10-genre audio, spec's "Easy baseline" for Tasks 1–2. No real captions exist, so captions are synthesized (see Limitations). |
| **DEAM** | Task 3 (real emotion regression), Task 4 | Real continuous valence/arousal (1–9 scale). No genre/tag labels exist natively — a 4-class mood quadrant (happy/energetic, calm/content, angry/tense, sad/melancholic) is derived from the real (valence, arousal) values, giving genuine tag + emotion targets from one real signal. |

### Preprocessing GTZAN
```bash
python src/preprocess_dataset.py --dataset gtzan \
    --raw-root data/raw/gtzan --out-root data/processed/gtzan
```

### Preprocessing DEAM
```bash
python src/preprocess_deam.py \
    --audio-root "data/raw/deam/MEMD_audio" \
    --annotations "data/raw/deam/annotations/annotations averaged per song/song_level" \
    --out-root data/processed/deam
```

### Training on real data
'''
python src\train.py --config src\config.yaml --task 1 --data-root data\processed --dataset gtzan --epochs 15
python src\train.py --config src\config.yaml --task 2 --data-root data\processed --dataset gtzan --epochs 30
python src\train.py --config src\config.yaml --task 3 --data-root data\processed --dataset gtzan --ablation --epochs 30
python src\train.py --config src\config.yaml --task 4 --data-root data\processed --dataset gtzan --epochs 30
'''

**Important:** every task writes to the same `results/metrics.json` under
fixed key names, regardless of dataset — back up `results/` (metrics.json,
plots/, retrieval_examples/) before switching from one dataset to another,
or the second run silently overwrites the first.

## Tasks implemented

| Task | Model | File | Metric |
|---|---|---|---|
| 1 — Easy | Text-encoder multi-label tag classifier | `bert_encoder.py` | Macro/Micro-F1, AUC-PR |
| 2 — Medium | GraphSAGE on segment graph | `gnn_model.py` | Macro-F1 vs CNN (B2) vs PCA+MLP (B4) |
| 3 — Hard | Cross-attention GNN-BERT fusion + emotion regression, 4-mode ablation | `fusion_model.py` | Macro-F1, AUC-PR, MAE, R², graph coherence score |
| 4 — Advanced | InfoNCE contrastive dual-encoder + zero-shot tag prediction | `contrastive.py` | R@1/5/10, zero-shot vs. Task 3 supervised comparison |



