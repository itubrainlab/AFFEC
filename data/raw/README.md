# Dataset: AFFEC - Advancing Face-to-Face Emotion Communication Dataset

## Overview
The **AFFEC (Advancing Face-to-Face Emotion Communication)** dataset is a multimodal dataset designed for emotion recognition research. It captures dynamic human interactions through **electroencephalography (EEG), eye-tracking, galvanic skin response (GSR), facial movements, and self-annotations**, enabling the study of **felt and perceived emotions** in real-world face-to-face interactions. The dataset comprises **84 simulated emotional dialogues**, **72 participants**, and **over 5,000 trials**, annotated with more than **20,000 emotion labels**.

> Repository policy: this devkit does **not** version raw dataset artifacts in git.
> Users should populate `data/raw/` locally using `python scripts/download_data.py`.

## Dataset Structure
The dataset follows the **Brain Imaging Data Structure (BIDS)** format and consists of the following components:

### Root Folder:
- `sub-*` : Individual subject folders (e.g., `sub-aerj`, `sub-mdl`, `sub-xx2`)
- `dataset_description.json`: General dataset metadata
- `participants.json` and `participants.tsv`: Participant demographics and attributes
- `task-fer_events.json`: Event annotations for the FER task
- `README.md`: This documentation file

### Subject Folders (`sub-<subject_id>`):
Each subject folder contains:
- **Behavioral Data (`beh/`):** Physiological recordings (eye tracking, GSR, facial analysis, cursor tracking) in JSON and TSV formats.
- **EEG Data (`eeg/`):** EEG recordings in `.edf` and corresponding metadata in `.json`.
- **Event Files (`*.tsv`):** Trial event data for the emotion recognition task.
- **Channel Descriptions (`*_channels.tsv`):** EEG channel information.

## Data Modalities and Channels

### 1. Eye Tracking Data
- **Channels:** 16 (fixation points, left/right eye gaze coordinates, gaze validity)
- **Sampling Rate:** 62 Hz
- **Trials:** 5632
- **File Example:** `sub-<subject>_task-fer_run-0_recording-gaze_physio.json`

### 2. Pupil Data
- **Channels:** 21 (pupil diameter, eye position, pupil validity flags)
- **Sampling Rate:** 149 Hz
- **Trials:** 5632
- **File Example:** `sub-<subject>_task-fer_run-0_recording-pupil_physio.json`

### 3. Cursor Tracking Data
- **Channels:** 4 (cursor X, cursor Y, cursor state)
- **Sampling Rate:** 62 Hz
- **Trials:** 5632
- **File Example:** `sub-<subject>_task-fer_run-0_recording-cursor_physio.json`

### 4. Face Analysis Data
- **Channels:** Over 200 (2D/3D facial landmarks, gaze detection, facial action units)
- **Sampling Rate:** 40 Hz
- **Trials:** 5680
- **File Example:** `sub-<subject>_task-fer_run-0_recording-videostream_physio.json`

### 5. Electrodermal Activity (EDA) and Physiological Sensors
- **Channels:** 40 (GSR, body temperature, accelerometer data)
- **Sampling Rate:** 50 Hz
- **Trials:** 5438
- **File Example:** `sub-<subject>_task-fer_run-0_recording-gsr_physio.json`

### 6. EEG Data
- **Channels:** 63 (EEG electrodes following the 10-20 placement scheme)
- **Sampling Rate:** 256 Hz
- **Reference:** Left earlobe
- **Trials:** 5632
- **File Example:** `sub-<subject>_task-fer_run-0_eeg.edf`

### 7. Self-Annotations
- **Trials:** 5807
- **Annotations Per Trial:** 4
- **Event Markers:** Onset time, duration, trial type, emotion labels
- **File Example:** `task-fer_events.json`

## Experimental Setup
Participants engaged in a **Facial Emotion Recognition (FER) task**, where they watched **emotionally expressive video stimuli** while their physiological and behavioral responses were recorded. Participants provided **self-reported ratings for both perceived and felt emotions**, differentiating between the emotions they believed the video conveyed and their internal affective experience.

The dataset enables the study of **individual differences in emotional perception and expression** by incorporating **Big Five personality trait assessments** and demographic variables.

## Usage Notes
- The dataset is formatted in **ASCII/UTF-8 encoding**.
- Each modality is stored in **JSON, TSV, or EDF format** as per **BIDS standards**.
- Researchers should **cite this dataset appropriately** in publications.

## Applications
AFFEC is well-suited for research in:
- **Affective Computing**
- **Human-Agent Interaction**
- **Emotion Recognition and Classification**
- **Multimodal Signal Processing**
- **Neuroscience and Cognitive Modeling**
- **Healthcare and Mental Health Monitoring**

## Acknowledgments
This dataset was collected with the support of **brAIn lab, IT University of Copenhagen**. Special thanks to all participants and research staff involved in data collection.

## Changelog

### v2 — April 2026: Videostream JSON Sidecar Corrections

**Issue:** In the original release (v1, December 2024), a subset of
`*_recording-videostream_physio.json` sidecar files contained a truncated
`Columns` list (e.g., 190 entries instead of 715) that did not match the
paired `*.tsv.gz` data file.  Downstream loaders that relied on the JSON
`Columns` field to parse column names would silently produce incorrect
feature matrices for the affected subjects and runs.

**Affected files:** 91 JSON sidecars across 24 subjects.

| Repair type | Subjects (n) | Files fixed |
|---|---|---|
| Standard (matched template from same subject or another run) | 18 | 69 |
| Forced surrogate template (TSV had only 3 header fields; full 715-column schema applied) | 6 | 22 |

**Standard-repaired subjects (18):**
`sub-rokb`, `sub-scuy`, `sub-srfl`, `sub-ssn`, `sub-tag`, `sub-tao`,
`sub-tico`, `sub-tqu`, `sub-ubc`, `sub-uwdm`, `sub-xx2`, `sub-xzc`,
`sub-yel`, `sub-ymjj`, `sub-ywh`, `sub-zig`, `sub-ziym`, `sub-zry`

**Forced-repaired subjects (6):**
`sub-agk`, `sub-bxn`, `sub-jgs`, `sub-ors`, `sub-rhn`, `sub-yarq`

**What changed:** Only the `Columns` array inside the sidecar `.json` file
was updated.  The binary `.tsv.gz` data files are **unchanged**.  Original
sidecar files are preserved as `*.json.bak` backups alongside the corrected
versions.

**Supplementary files added in v2:**
- `videostream_corrections_only.zip` — patch archive containing only the
  91 corrected JSON sidecars (4 MB).  Users of v1 can apply this patch
  without re-downloading the full `videostream.zip`.
- `videostream_json_repair_report.csv` — per-file status for the standard
  repair pass.
- `videostream_json_repair_report_forced.csv` — per-file status for the
  forced repair pass.
- `videostream_unresolved_runs.csv` — records of runs where the TSV itself
  contained only 3 header fields and required surrogate-template repair.

**Devkit packaging path:** Rebuilt upload artifacts are generated under
`data/zenodo_upload/` by `scripts/prepare_zenodo_upload.py`.

## License

This dataset is shared under the [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license.

## Citation

Please cite this dataset if you use it in your research:

```
[Citation to be added upon publication]
```

## Contact

For questions or collaboration inquiries, open an issue in the
[AFFEC devkit repository](https://github.com/brAInlab-ITU/AFFEC) or contact
the brAIn lab, IT University of Copenhagen.

