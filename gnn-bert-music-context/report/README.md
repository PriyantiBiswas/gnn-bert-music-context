# Final Report

Place your NeurIPS/IEEE/ICML-style report PDF here as `final_report.pdf`
(6-10 pages, per Section 10 of the project spec).

Overleaf templates:
- NeurIPS 2024: https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh
- IEEE Conference: https://www.overleaf.com/latex/templates/ieee-conference-template
- ICML 2025: https://www.overleaf.com/latex/templates/icml2025-template

## Suggested report outline (maps to the rubric in Table 4)

1. **Introduction & motivation** — musical context, why BERT+GNN fusion
2. **Related work** — BERT text encoders, GraphSAGE/GAT, contrastive learning (CLIP/CLAP-style)
3. **Method**
   - Problem definition (Section 2 of spec)
   - Preprocessing pipeline (Section 3): audio → mel/chroma → segment/chord graphs → BERT tokens
   - Task 1-4 architectures and losses (Section 4)
4. **Experiments**
   - Datasets & splits (no artist leakage)
   - Baselines: B1 (majority/random), B2 (CNN mel-spec), B3 (BERT-only), B4 (PCA+MLP)
   - Results tables: Macro-F1 / Micro-F1 / AUC-PR / MAE / R@K — see `results/metrics.json`
     and `results/summary.json`
   - Ablations (Task 3): BERT-only vs GNN-only vs early-concat vs cross-attention
   - Figures: `results/plots/loss_curves.png`, `task1_f1_vs_epoch.png`, `task3_tsne.png`,
     `baseline_comparison.png`
5. **Case studies** — 3 examples showing graph paths + caption/lyric alignment (Task 3);
   10 qualitative retrieval examples (`results/retrieval_examples/`)
6. **Discussion & limitations**
7. **Conclusion**

## Note on the reference implementation in this repo

The shipped code trains and evaluates end-to-end on a *synthetic* dataset
(`--synthetic` flag) so the pipeline is fully runnable without downloads.
The synthetic generator intentionally makes graph features, tags, and
genre mutually informative (see `src/datasets.py`), so metrics reported by
`make_results.py` demonstrate the pipeline is learning correctly — they are
**not** to be reported as final results. Regenerate all tables/plots on a
real dataset (FMA-medium / MagnaTagATune / MusicCaps / DEAM per Table 1)
before writing the report, using `train.py --dataset <name>` once the
corresponding loader in `datasets.py` is filled in and real data is cached
under `data/processed/`.
