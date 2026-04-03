# AFFEC Devkit

Developer toolkit for the **AFFEC (Advancing Face-to-Face Emotion Communication)** dataset.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14794876.svg)](https://doi.org/10.5281/zenodo.14794876)
[![License: CC0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

AFFEC is a multimodal affective computing dataset capturing **72 participants** across **84 simulated emotional dialogues** (~5 000 trials, 20 000+ emotion labels). Modalities: **EEG, Eye Tracking, Pupil, GSR, Facial Action Units, Cursor Tracking**, with **felt and perceived** arousal/valence labels.

---

## What this repo is

| Component | Description |
|-----------|-------------|
| `affec/` | Python package — data loading, feature extraction, model wrappers, utilities |
| `scripts/` | CLI scripts: download data, run full analysis, statistical analysis, prepare Zenodo upload |
| `multimodal_emotion_recogntion.ipynb` | Full pipeline notebook (feature extraction → AutoGluon → 5-fold CV → results table) |
| `demo_analysis.ipynb` | Shorter demo showing end-to-end workflow |
| `data/raw/README.md` | Dataset schema, modality specs, changelog (v2 JSON sidecar corrections) |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python scripts/download_data.py
```

This downloads all modality zips from Zenodo ([DOI 10.5281/zenodo.14794876](https://doi.org/10.5281/zenodo.14794876)) into `data/raw/`.

### 3. Run the pipeline

**Interactive (recommended):**

```bash
jupyter notebook demo_analysis.ipynb
```

Run notebook cells from top to bottom to initialize shared variables (`DATA_DIR`, `Config`, imports) before running analysis cells.

**Full analysis:**

```bash
python scripts/run_full_analysis.py
python scripts/run_statistical_analysis.py
```

**Quick validation (8 participants):**

```bash
python scripts/run_full_analysis.py --all-targets --max-participants 8
python scripts/run_statistical_analysis.py --max-participants 8
```

---

## Modality configuration

Toggle modalities in the notebook or pass them programmatically:

```python
USE_MODALITIES = {
    "eye":           True,   # Eye tracking (gaze, fixation, pupil dilation)
    "action_units":  True,   # Facial Action Units (AU)
    "gsr":           True,   # Galvanic Skin Response
    "personality":   True    # Big Five personality traits (OCEAN)
}
```

| Configuration | Eye | AU | GSR | Personality |
|---|:---:|:---:|:---:|:---:|
| Full multimodal | ✅ | ✅ | ✅ | ✅ |
| Eye only | ✅ | ❌ | ❌ | ❌ |
| AU only | ❌ | ✅ | ❌ | ❌ |
| GSR only | ❌ | ❌ | ✅ | ❌ |
| No personality | ✅ | ✅ | ✅ | ❌ |

---

## Baseline results (5-fold CV, full multimodal)

| Metric | Perceived Arousal | Perceived Valence | Felt Arousal | Felt Valence |
|--------|:-----------------:|:-----------------:|:------------:|:------------:|
| Best model | XGBoost | XGBoost | LightGBMXT | NeuralNetFastAI |
| High F1 | 0.457 ± 0.016 | 0.232 ± 0.025 | 0.269 ± 0.040 | 0.473 ± 0.030 |
| Medium F1 | 0.362 ± 0.017 | 0.431 ± 0.017 | 0.485 ± 0.025 | 0.294 ± 0.027 |
| Low F1 | 0.495 ± 0.021 | 0.610 ± 0.015 | 0.680 ± 0.016 | 0.613 ± 0.025 |
| Macro F1 | 0.438 ± 0.008 | 0.424 ± 0.015 | 0.478 ± 0.014 | 0.460 ± 0.019 |
| Accuracy | 0.442 ± 0.009 | 0.506 ± 0.015 | 0.568 ± 0.016 | 0.514 ± 0.022 |

---

## Dataset

See [`data/raw/README.md`](data/raw/README.md) for full schema, modality specs, and the v2 changelog (April 2026 videostream JSON sidecar corrections).

**Download:** [https://doi.org/10.5281/zenodo.14794876](https://doi.org/10.5281/zenodo.14794876)

### Data policy (important)

This repository is a **data-free devkit**. Raw AFFEC files are not committed to git.

- Download data only via [scripts/download_data.py](scripts/download_data.py)
- Store downloaded files under `data/raw/` locally
- Do not commit any files under `data/` except [data/raw/README.md](data/raw/README.md)

To reconstruct the local dataset layout:

```bash
python scripts/download_data.py
```

---

## Local-only folders in this devkit

- `docs/` and `logs/` are treated as local working material.
- `data/zenodo_upload/` is local packaging output for Zenodo updates.
- Large raw data artifacts (`data/raw/sub-*`, modality `.zip` files, `.bak`) are ignored.

---

## Requirements

- Python ≥ 3.8
- AutoGluon, Pandas, NumPy, scikit-learn, NeuroKit2, Tabulate

```bash
pip install -r requirements.txt
```

---

## Citation

If you use AFFEC in your research, please cite:

```
[Citation to be added upon publication]
```

---

## License

Dataset: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)  
Code: MIT ([LICENSE](LICENSE))
