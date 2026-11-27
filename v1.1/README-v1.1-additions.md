# Additions in release v1.1.0

These files extend release v1.0.0 for the revised manuscript (600 models in total). Nothing in v1.0.0 is changed.

## New benchmark: NF-CSE-CIC-IDS2018-v3
- `protocols/cic18_turnover_v1.protocol.json` – preparation protocol, frozen before any model was trained
  (periods, label-free client hash, sampling caps, root draws).
- `protocols/cic18-sampling-cells.json` – eligible and sampled rows per cell (gives the sampling weights).
- `protocols/manifest-base.json`, `protocols/preprocessing.json` – manifest and feature transform of the prepared arrays.
- `labels/cic18/{cal,period_a,period_b,seen}.npz` – label, family, sampling weight, client, internal host, day,
  start time and source row index; **no features**. `seen` is the held-out seen-family set.
- `scores/cic18-panel/` – 90 models (FLTrust, trimmed mean, FedAvg); `scores.npz` holds round-600 scores for
  `cal`, `period_a`, `period_b` and `seen`; `result.json` is the slim training record.
- `code/audit_cic18.py`, `code/prepare_cic18_turnover_v1.py` – family audit and three-stage preparation
  (requires the public NF-CSE-CIC-IDS2018-v3 archive, DOI 10.48610/ece9b83).

## Rule and architecture extension
- `code/rules_arch_runner_v2.py` – copy of `cross_testbed_runner_v1.py` that adds coordinate-wise median,
  Krum (f = 1) and architecture B (hidden widths 256-128-64); nothing else changes.
- `scores/rules-arch-extension/` – 180 models: on the NF-ToN-IoT-v3 matched cells, median and Krum
  (architecture A) and FLTrust and trimmed mean (architecture B); on NF-CSE-CIC-IDS2018-v3, median and Krum.
  NF-ToN-IoT-v3 labels are in `labels/` of v1.0.0 (look up `input_sha256` in each `result.json`).
- `code/srv_fedids_cic18.py`, `code/srv_fedids_rules.py` – the job launchers used on the GPU server.

## Analysis
- `code/generality_analysis.py` – `python3 generality_analysis.py cic18|nf`; reproduces Section 4.3 and,
  on the original NF-ToN-IoT-v3 panel, the matched-panel numbers of Sections 4.1–4.2. Paths resolve from this package; see the commands below.
- `results/cic18-generality.json`, `results/nf-generality.json`, `results/cic18-family-audit.json`.

## Portable paths and running the analyses

The release layout is `<repository>/v1.1/`, next to the unchanged v1.0 label and score directories.
Run from the repository root with the original package dependencies installed:

```sh
python3 v1.1/code/generality_analysis.py cic18
python3 v1.1/code/generality_analysis.py nf
```

The scripts load new labels and scores relative to their own location. NF analyses load the existing
v1.0 `labels/` and `scores/` from the repository root. `FEDIDS_V1_ROOT` can override that root.
Outputs go to `v1.1/outputs/`, or to `FEDIDS_ANALYSIS_OUT` when supplied. Archived `results/` are not overwritten.
The analysis uses NumPy; dataset audit and preparation additionally need pandas. Training needs PyTorch.

The preparation script resolves its frozen protocol from `protocols/`; its command-line data and output
arguments are unchanged. The `/data/NF-CICIDS2018-v3.csv` string matches a member inside the original ZIP,
not a local absolute path. Original archived protocol and preparer hashes remain provenance of the original
experiments; path-adapted source files have new hashes in the repository manifest.

The two `srv_fedids_*.py` scripts are optional historical launchers for a Linux GPU environment, not analysis
entry points. They can start training jobs when run and require explicit local environment paths:
`FEDIDS_WORK_DIR`, `FEDIDS_TRAIN_PYTHON`, plus `FEDIDS_CROSS_TESTBED_DIR` for the CIC18 launcher or
`FEDIDS_NF_INPUT_DIR` for the rule launcher. They do not contain server connection details. They were not
executed during release preparation. Feature generation is available as source but no feature matrices are distributed.
