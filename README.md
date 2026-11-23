# Label noise in validation data and threshold calibration of ML network intrusion detectors

Code and per-seed results for the paper

> R. U. Zaman, G. R. Patil, S. H. Begum, F. Taranum, M. A. Raheem and G. Suhasini, "Label Noise in Validation Data and a
> Contamination-Based Recall Bound for Threshold Calibration of ML Network Intrusion Detectors", submitted to the
> International Journal of Intelligent Systems and Applications (IJISA).

The study injects four kinds of label noise (symmetric flips, attacks labeled benign, benign records labeled attack, and one
attack family partly labeled benign) into the training labels, the validation labels or both, trains five detectors
(logistic regression, random forest, XGBoost, LightGBM, Isolation Forest), sets the threshold for a 1% false-positive rate
from the validation records labeled benign, and reports recall on a control set whose labels are never changed. Ten seeds
(42 to 51) are used throughout, and every comparison is paired by seed.

## Contents

| Path | What it is |
|---|---|
| `code/P9_run_v1.0.py` | Main study on NSL-KDD and UNSW-NB15 (all noise conditions, confident-learning cleaner, remedies, statistics). |
| `code/P9_run_v1.1.py` | Main study extended to CICIDS2017 (cleaned release). |
| `code/P9_run_v2.0.py` | Revision experiments: `--part select` (score-dependent noise in the validation labels), `--part cleaners` (confident learning, ensemble filter, co-teaching), `--part full` (the whole study on CICIoT2023), `--part tuned` (grid-searched detectors). |
| `code/P9_select_diag.py` | Diagnostic of the `select` part: position and family make-up of the mislabeled validation attacks and the generalized bound. |
| `code/analysis_v2.py` | Tables and tests of the revision (paired Wilcoxon tests with Holm correction). |
| `results/v1.0/seed_*/` | Per-seed results of the main study, NSL-KDD and UNSW-NB15 (`results.csv`: one row per dataset, condition, remedy and detector; `cleaner.csv`: cleaner accuracy). |
| `results/v1.1/seed_*/` | The same for CICIDS2017. |
| `results/v2.0/<part>/seed_*/` | Per-seed results of the revision experiments. |
| `results/v2.0/tuned_params.json`, `tuning_grid.csv` | Grid-search results and the chosen settings. |
| `results/v2.0/hardware.json` | Hardware and library versions of the revision runs. |

## Data

The datasets are not redistributed. Download them from their providers:

- NSL-KDD: `KDDTrain+.txt`, `KDDTest+.txt` (Canadian Institute for Cybersecurity, University of New Brunswick).
- UNSW-NB15: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` (UNSW Canberra).
- CICIDS2017: the cleaned release `cicids2017_cleaned.csv` by E. A. Ribeiro on Kaggle (ericanacletoribeiro/cicids2017-cleaned-and-preprocessed).
- CICIoT2023: `CSV.zip` from the Canadian Institute for Cybersecurity. `P9_run_v2.0.py` reads every CSV file in the archive and
  keeps a fixed random 5% of each benign file and 0.05% of each attack file (seeded), so the sample is rebuilt exactly.

Put the first three under `<data-dir>/p12/` and `CSV.zip` under `<data-dir>/iot/ciciot2023/`.

## Running

Python 3.11 with numpy, pandas, scikit-learn, xgboost, lightgbm, scipy and (for co-teaching) torch. The revision runs used
the versions in `results/v2.0/hardware.json`.

```
python code/P9_run_v2.0.py --part full     --datasets CICIoT2023 --data-dir <data-dir> --out-dir out/p9_v2
python code/P9_run_v2.0.py --part select   --data-dir <data-dir> --out-dir out/p9_v2
python code/P9_run_v2.0.py --part cleaners --data-dir <data-dir> --out-dir out/p9_v2
python code/P9_run_v2.0.py --part tuned    --data-dir <data-dir> --out-dir out/p9_v2
python code/P9_select_diag.py --data-dir <data-dir> --out out/p9_v2/select_diag
```

Every part is restartable: a seed whose folder holds `DONE.json` is skipped. `--quick` runs a small smoke test. The v1.0 and
v1.1 scripts find the data folder themselves (see the header of each script for the expected layout).

## License

MIT License (see `LICENSE`).
