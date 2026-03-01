# NeuroCommitSSM  (Notebooks & Pipeline)

This repository contains the end-to-end notebook pipeline for our IROS paper, including data preprocessing (Phase 1B → Phase 5.5) and the model notebook for training/evaluation.

---


## Repository layout

```text
Artifacts/
  final_cfg.json
  final_model.pt
  final_thresholds.json
  stats_fold.json

Data preprocessing/
  Phase-1B.ipynb
  Phase-1C.ipynb
  Phase-2A.ipynb
  Phase-2B.ipynb
  Phase-3.ipynb
  Phase-4A.ipynb
  Phase-4B.ipynb
  Phase-5A.ipynb
  Phase-5B.ipynb   # includes/ends with Phase 5.5 feature extraction

Model Notebook/
  Model.ipynb 

```

---


### 1) Data preprocessing (run in order: Phase 1B → Phase 5.5)

Inside **Data preprocessing/**, run notebooks **sequentially**:

1. **Phase-1B.ipynb** — Tri-modal synchronization/alignment (EEG–EMG–Eye Tracking)
2. **Phase-1C.ipynb** — Project-specific cleanup/verification/export steps
3. **Phase-2A.ipynb** — Labeling / onset–offset detection logic
4. **Phase-2B.ipynb** — Label audits, fixes, exports
5. **Phase-3.ipynb** — Manifest + TRUE-LOSO splits + windowing
6. **Phase-4A.ipynb** — Deterministic preprocessing (filters, envelopes, masks) → cached arrays
7. **Phase-4B.ipynb** — QC summaries (recommended)
8. **Phase-5A.ipynb** — Fold-wise export (balanced supervised + SSL/unbalanced)
9. **Phase-5B.ipynb** — **Phase 5.5 feature extraction** (features used by the model)


---

### 2) Model Notebook

Go to **Model Notebook/Model.ipynb** and run it to:
- train and/or fine-tune the model
- evaluate under sensor-dropout scenarios (as configured in the notebook)
- save metrics/artifacts /update files in **Artifacts/**

---

## Artifacts (final model bundle)

The **Artifacts/** folder contains a ready-to-use model bundle:

- **final_model.pt** — trained PyTorch weights
- **final_cfg.json** — model + data/config settings used for training/inference
- **final_thresholds.json** — tuned thresholds (e.g., action/commit gating as used in the paper)
- **stats_fold.json** — normalization statistics (recommended for consistent inference)

**High-level inference requirements**
1) Load **final_model.pt**
2) Apply the same preprocessing/features as Phase 4 → Phase 5.5
3) Normalize using **stats_fold.json**
4) Apply thresholds from **final_thresholds.json**

See **Model Notebook/Model.ipynb** for the exact loading/evaluation code.

---

## Dataset

The dataset will be available publicly once the paper is accepted.

---

## Environment

- Python **3.10+**
- Jupyter Notebook/Lab
- Recommended: CUDA-enabled PyTorch for training

Common dependencies:
- numpy, pandas, scipy, scikit-learn, matplotlib, tqdm
- torch
- (optional) mne

---

## Citation

 
**We will replace the placeholders after acceptance.**

```bibtex
@inproceedings{NeuroCommitSSM_IROS_2026,
  title     = {NeuroCommitSSM: TODO (paper title after acceptance)},
  author    = {TODO: Author list},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}

---


