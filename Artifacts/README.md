# NeuroCommitSSM Inference

This folder contains the inference artifacts for **NeuroCommitSSM**, including the trained model, configuration files, calibrated thresholds, and the Python script required to run prediction on a new preprocessed trial.

## Overview

NeuroCommitSSM is used here in **inference mode** to generate predictions from a single preprocessed trial file. The script loads the trained model, applies the saved configuration and thresholds, processes the input `.npz` file, and writes the prediction results to a CSV file.

## Required Files

The following files are required to run inference:

- `final_model.pt` — trained model weights  
- `final_cfg.json` — model configuration  
- `final_thresholds.json` — calibrated thresholds for inference  
- `stats_fold.json` — normalization/statistics file  
- `neurocommit_model.py` — inference script  

## Input

The model expects a **preprocessed `.npz` file** as input.

Example input file:

```bash
C:\Users\tsultan1\Downloads\Test\New_data\Sub-38\cleaned\synchronized_proper_lite_union_v3\label\075_T18_synchronized_corrected_icml_consensus_labels.csv.preproc.v4.3.npz
```

## Output

The script generates a prediction CSV file as output.

Example output file:

```bash
C:\Users\tsultan1\Downloads\Test\TrialPreds\075_T18_pred.csv
```

## Requirements

Install the required Python packages before running the model:

```bash
pip install torch numpy pandas scikit-learn
```

## Example Folder Structure

```text
Test/
├── Model/
│   ├── final_cfg.json
│   ├── final_model.pt
│   ├── final_thresholds.json
│   └── neurocommit_model.py
├── stats_fold.json
├── New_data/
│   └── Sub-38/
│       └── cleaned/
│           └── synchronized_proper_lite_union_v3/
│               └── label/
│                   └── 075_T18_synchronized_corrected_icml_consensus_labels.csv.preproc.v4.3.npz
└── TrialPreds/
```

## How to Run

Run the following command in **Windows Command Prompt**:

```bash
py neurocommit_model.py ^
  --model_dir "C:\Users\tsultan1\Downloads\Test\Model" ^
  --data_root "C:\Users\tsultan1\Downloads\Test\New_data\Sub-38\cleaned\synchronized_proper_lite_union_v3\label\075_T18_synchronized_corrected_icml_consensus_labels.csv.preproc.v4.3.npz" ^
  --stats "C:\Users\tsultan1\Downloads\Test\stats_fold.json" ^
  --scenario S0 ^
  --out_csv "C:\Users\tsultan1\Downloads\Test\TrialPreds\075_T18_pred.csv" ^
  --device cpu
```

## Argument Description

### `--model_dir`

Path to the folder containing the model files:

- `final_model.pt`
- `final_cfg.json`
- `final_thresholds.json`

Example:

```bash
--model_dir "C:\Users\tsultan1\Downloads\Test\Model"
```

### `--data_root`

Path to the preprocessed input `.npz` file for one trial.

Example:

```bash
--data_root "C:\Users\tsultan1\Downloads\Test\New_data\Sub-38\cleaned\synchronized_proper_lite_union_v3\label\075_T18_synchronized_corrected_icml_consensus_labels.csv.preproc.v4.3.npz"
```

### `--stats`

Path to the statistics file used for normalization during inference.

Example:

```bash
--stats "C:\Users\tsultan1\Downloads\Test\stats_fold.json"
```

### `--scenario`

Inference scenario to use.

Example:

```bash
--scenario S0
```

### `--out_csv`

Path where the prediction CSV file will be saved.

Example:

```bash
--out_csv "C:\Users\tsultan1\Downloads\Test\TrialPreds\075_T18_pred.csv"
```

### `--device`

Device used for inference:

- `cpu`
- `cuda` (if GPU is available and PyTorch CUDA is installed)

Example:

```bash
--device cpu
```

## Full Example Command

```bash
py neurocommit_model.py ^
  --model_dir "C:\Users\tsultan1\Downloads\Test\Model" ^
  --data_root "C:\Users\tsultan1\Downloads\Test\New_data\Sub-38\cleaned\synchronized_proper_lite_union_v3\label\075_T18_synchronized_corrected_icml_consensus_labels.csv.preproc.v4.3.npz" ^
  --stats "C:\Users\tsultan1\Downloads\Test\stats_fold.json" ^
  --scenario S0 ^
  --out_csv "C:\Users\tsultan1\Downloads\Test\TrialPreds\075_T18_pred.csv" ^
  --device cpu
```

## Notes

- Make sure the input file is a valid preprocessed `.npz` file compatible with the trained NeuroCommitSSM model.
- Ensure that `stats_fold.json` matches the model used for inference.
- The `scenario` argument should be set according to the intended evaluation or deployment condition.
- If you want to use GPU inference, replace `--device cpu` with `--device cuda` only if CUDA is available on your system.



## Summary

To run inference with NeuroCommitSSM, you need:

1. The model artifact files in the `Model` folder  
2. A valid preprocessed `.npz` input file  
3. The `stats_fold.json` file  
4. The output path for saving predictions  

After that, run the command shown above and the script will generate the prediction CSV file for the given trial.
