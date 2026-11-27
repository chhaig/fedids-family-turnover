# Attack-family turnover confounds chronological comparisons of robust federated intrusion detectors at low false positive rates

**Author:** Haiguang Chen, Shanghai Normal University  
**Contact:** chhg@shnu.edu.cn

This repository contains the paper's analysis and training code, frozen protocols,
evaluation label metadata, saved model scores, and numerical analysis results.
The four original datasets must be downloaded from their official sources:

- **NF-ToN-IoT-v3** and **NF-UNSW-NB15-v3**: [University of Queensland NIDS datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/).
- **CIC-IDS2017**: [Canadian Institute for Cybersecurity, University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html).
- **IoT-23**: [Stratosphere Laboratory](https://www.stratosphereips.org/datasets-iot23).

**Original datasets and prepared feature matrices are not redistributed.**
Dataset endpoint IP addresses are research labels, not repository infrastructure.
Follow the original dataset providers' terms and citation requirements.

## Contents

- `code/`: 19 original analysis, training, and local orchestration scripts, with
  portable paths adapted for this package. No server connection is required for score analysis.
- `labels/`: 19 NPZ evaluation metadata packages, `index.json`, and
  `train_families.json`. Metadata include labels, sampling weights, attack families,
  endpoints, dates/timestamps, row indices, and input hashes; no feature arrays.
  NPZ filenames use the first 16 hex digits of prepared input SHA-256 hashes.
- `scores/`: step-600 calibration, earlier-period, and later-period scores and
  per-model `result.json` across six panels. Arrays are `score`, `fixed`, and `historical`.
- `protocols/`: four frozen training protocols. Original infrastructure identifiers
  are replaced with neutral worker labels; historical source/artifact hashes are
  provenance records, not checksums of the path-adapted public code.
- `results/`: the archived analysis outputs used for the paper. These are reference
  results; run analyses in a separate directory to avoid overwriting them.
- `MANIFEST.sha256.json`: SHA-256 of each distributed file except the manifest itself.

## Environment

Use Python 3.10 or later for analysis:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements are compatibility ranges, not an exact environment lock.
Original per-run Python, NumPy, PyTorch, CUDA, and GPU information is preserved
in `scores/*/*/result.json`. Training also requires a compatible PyTorch/CUDA
installation, prepared features, partitions, and original source data.

## Reproduce analyses from the included scores

Run the following from the repository root. Scripts locate `scores/` and `labels/`
relative to their own file location; output filenames remain relative to the
working directory. All outputs below are written into ignored `generated/`.

```sh
mkdir -p generated
cd generated
python ../code/unified_rescore.py
python ../code/pool_swap.py
python ../code/summarize.py
python ../code/bootstrap_pairs.py
python ../code/family_decomp.py
python ../code/op_sensitivity.py
python ../code/seed_replication.py
python ../code/three_rule.py
python ../code/seen_family_and_stability.py
python ../code/endpoint.py
python ../code/figs.py
python ../code/fig_family_mix.py
```

`unified_rescore.py` writes `unified_rescore.json`; its archived reference is named
`results/unified_rescore_panels.json`. It checks recalculated historical metrics
against the saved per-model metrics. The public score tree has one directory per
model; archived results may retain duplicate worker records. Compare by
`(matrix, run)` rather than raw list length or order.

`bootstrap_pairs.py` defaults to 400 replicates and resumes an existing output;
remove only `generated/bootstrap_pairs.json` before starting a fresh bootstrap.
The bootstrap can take substantial time. All statistical formulas, thresholds,
seeds, and iteration counts in the supplied analysis code are preserved.

The same-day feature-based control cannot be rerun from scores alone. Its
archived output is `results/same_day_family.json`. To rerun, obtain or reconstruct
the prepared development features and frozen model weights, set `FEDIDS_DEV`
to the development NPZ (default `external-data/cross_dev.npz`), and supply the
model-panel directories containing `model-final.pt` and `result.json`:

```sh
# From generated/, after supplying the external inputs:
python ../code/same_day_family.py ../external-workspace/model-panel
```

## Training and metadata-export code

`cross_testbed_runner_v1.py` and `nf_trajectory_runner_v1.py` expose their required
local inputs through `--help`. Four `srv_fedids_*.py` files preserve the original
local job orchestration and metadata export logic with relative workspace paths.
They do not connect to a server. They require the prepared training workspace
under `external-workspace/` and write only their relative `local-work/` outputs.
The seed orchestration script uses the current Python interpreter, which must
have a working CUDA-enabled PyTorch installation. These utilities are not part
of the score-only analysis procedure. Frozen protocols and runners alone do not
constitute a complete raw-data preparation pipeline; no claim is made that a
fresh training run is possible without the external prepared artifacts.

## Integrity

From the repository root:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('MANIFEST.sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print(f'Verified {len(manifest)} files')
PY
```

## License

Code is licensed under the **MIT License**; see `LICENSE`.
Scores, label metadata, protocols, and result files are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Attribute Haiguang Chen and cite the paper and this repository. This grant does
not replace the original dataset providers' terms for their source datasets.

## Citation

Haiguang Chen. *Attack-family turnover confounds chronological comparisons of
robust federated intrusion detectors at low false positive rates*. Manuscript;
DOI will be added after acceptance.

```bibtex
@unpublished{chen_fedids_family_turnover,
  author = {Chen, Haiguang},
  title = {Attack-family turnover confounds chronological comparisons of robust
           federated intrusion detectors at low false positive rates},
  note = {Manuscript. Reproducibility package: fedids-family-turnover.
          DOI to be added after acceptance.}
}
```


## v1.1.0 additions (local preparation; not yet released)

The `v1.1/` directory adds NF-CSE-CIC-IDS2018-v3 (90 models) and a rule/architecture
extension (180 models), bringing the combined archive to 600 models. It includes four CIC18
evaluation-label metadata files without features, 270 compact score sets, six source scripts,
four preparation/protocol JSON files and three analysis-result JSON files.

The extension evaluates coordinate-wise median and Krum (f=1), plus FLTrust and trimmed mean
with architecture B (256-128-64). New analyses cover family turnover, pool exchange, family
mixture decomposition, mix stability, seen-family controls and threshold transfer.

See [the v1.1 additions guide](v1.1/README-v1.1-additions.md) for contents, dependencies and commands.
The archived v1.0.0 code, labels, scores, protocols and numerical results remain unchanged;
only this README and the aggregate integrity manifest are updated to describe the addition.
No original datasets, feature matrices or model weights are added.
