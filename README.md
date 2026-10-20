# fMRI Movie Watching × MLLM Ratings

Multimodal LLMs (Gemma, Qwen3-VL, LLaDA) rate Theory-of-Mind (ToM) and valence for three movies:
The Present (TP), Despicable Me (DM) and Partly Cloudy (PC). We compare those ratings with human
ratings and use both to predict fMRI responses with encoding models.

## Layout

| Path | Contents |
|---|---|
| `rating_generation/` | MLLM rating and description scripts. The **prompts** are defined inside these scripts, and `context_variants.py` holds the context-window variants. |
| `run_context_ablation*.sh` | Context-window ablation runs (4/8/12/16 s) |
| `rating_utils.py` | Shared helpers: load ratings, resample to the fMRI TR grid, smooth |
| `rating_benchmark/` | Model-vs-human rating metrics, best lag, inter-model agreement, context ablation, calibration |
| `rating_human_vs_machine/` | Model-vs-human rating correlations and ToM/valence time-series plots |
| `encoding_pipeline copy.py` | Encoding-model pipeline (banded ridge, FIR delays, GroupKFold) |
| `build_features_human_vs_model.py` | Human and model rating features on the fMRI grid |
| `run_hvm_diagnostic_singledelay.{py,sh}` | Separate (solo) and joint encoding of human and model ratings with a single delay |
| `analyze_*solo*.py`, `paired_source_tests.py` | ROI significance, human/model overlap, paired source tests |
| `plot_solo_overlap_count_brains.py`, `plot_solo_heatmap_v2.py` | Main brain-map and heatmap figures |

## Not included

This repo contains code and prompts only. Data and results stay local: fMRI arrays, movie files, human ratings, MLLM rating outputs (`embedding/`), and all result arrays and figures.
