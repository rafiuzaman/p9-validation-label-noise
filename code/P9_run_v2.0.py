"""Paper 9 v2.0 - additional experiments for the IJISA revision (PAPER024943), Nov 2026.

Parts (run separately; each is restartable, finished seeds are skipped):
  select    attacks in the validation set labeled benign at random or depending on how hard they are (reviewer M1)
  tuned     all noise conditions with detectors tuned per dataset (M5)
  cleaners  confident learning vs an ensemble filter and co-teaching (M2)
  full      the v1.1 study on CICIoT2023 (M4)

Example (VM 22):
  python P9_run_v2.0.py --part select --data-dir /mnt/hgfs/gdrive/data --out-dir /mnt/hgfs/gdrive/vm22/results/p9_v2
The data, noise model, detectors, thresholds and seeds are those of v1.1; only the parts above are new.
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ configuration
SEEDS = list(range(42, 52))
MODELS = ["LR", "RF", "XGB", "LGBM", "IF"]
SUPERVISED = ["LR", "RF", "XGB", "LGBM"]
MODEL_LABELS = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost",
                "LGBM": "LightGBM", "IF": "Isolation Forest"}
ALPHA = 0.01
HOLDOUT_FRAC = 0.20
TRAIN_FRAC = 0.50
SPLIT_SEED = 2026
FAMILY_RATE = 0.50
REMEDIES = ["none", "remove", "relabel"]
VERSION = "v1.1"
ALL_DATASETS = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
CIC_TRAIN_N = 100000      # per-seed training sample (NSL-KDD uses ~50k, UNSW-NB15 ~40k)
CIC_HOLDOUT_N = 100000    # per-seed held-out sample, split 50/50 into V and C
MIN_FAMILY = 500          # family-noise condition only if the family has this many training records
CIC_BENIGN = ("normal traffic", "benign", "normal")

NSL_CAT = ['protocol_type', 'service', 'flag']
UNSW_CAT = ['proto', 'service', 'state']
NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
# Standard mapping of the attack names that occur in KDDTrain+ to the four NSL-KDD categories.
NSL_FAMILY = {
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
}

MODEL_COLORS = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a", "LGBM": "#8a5cc7", "IF": "#c9a227"}
REMEDY_COLORS = {"none": "#6b6a66", "remove": "#2a78d6", "relabel": "#eb6834"}
INK, INK_2, BASELINE_COLOR = "#0b0b0b", "#52514e", "#6b6a66"

log = logging.getLogger("P9")
QUICK = False
FALLBACK = False          # only for testing without xgboost/lightgbm installed


# ------------------------------------------------------------------ helpers
def setup_logging(out_dir):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(out_dir, "run_log.txt"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.handlers = [fh, sh]


def keep_awake(on):
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS)
        return True
    except Exception:
        return False


def find_papers_dir(cli_value):
    candidates = [cli_value] if cli_value else [
        r"H:\My Drive\Publish\Papers", r"G:\My Drive\Publish\Papers", "/content/drive/MyDrive/Publish/Papers"]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "Paper6")):
            return c
    sys.exit("Could not find the Papers folder (looked for Paper6 inside: %s). Use --papers-dir." % ", ".join(map(str, candidates)))


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    sys.exit("File not found; looked for: " + " | ".join(paths))


def strat_sample(y, n, seed, pool=None):
    from sklearn.model_selection import train_test_split
    pool = np.arange(len(y)) if pool is None else pool
    if n >= len(pool):
        return pool
    try:
        idx, _ = train_test_split(pool, train_size=n, stratify=y[pool], random_state=seed)
    except ValueError:     # a stratum too small to split (only possible for CICIDS2017 rare types)
        idx, _ = train_test_split(pool, train_size=n, random_state=seed)
    return np.sort(idx)


# ------------------------------------------------------------------ data
def read_nsl(path):
    df = pd.read_csv(path, names=NSL_COLUMNS)
    lab = df["label"].astype(str).str.strip()
    df["y"] = (lab != "normal").astype(int)
    df["family"] = np.where(df["y"] == 0, "Normal", lab.map(NSL_FAMILY).fillna("Other"))
    return df.drop(columns=["label", "difficulty"])


def read_unsw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["y"] = df["label"].astype(int)
    df["family"] = np.where(df["y"] == 0, "Normal", df["attack_cat"].astype(str).str.strip())
    return df.drop(columns=["id", "attack_cat", "label"])


def encode_categoricals(train, test, cat_cols):
    for col in cat_cols:
        codes = {v: i for i, v in enumerate(sorted(train[col].astype(str).unique()))}
        train[col] = train[col].astype(str).map(codes).astype(int)
        test[col] = test[col].astype(str).map(codes).fillna(-1).astype(int)
    return train, test


def prepare(name, train_df, test_df, cat_cols):
    from sklearn.model_selection import train_test_split
    # duplicates judged on features + binary label (as in P12/P13); the family column is not part of the key
    key = [c for c in train_df.columns if c != "family"]
    train_df = train_df.drop_duplicates(subset=key).reset_index(drop=True)
    test_df = test_df.drop_duplicates(subset=key).reset_index(drop=True)
    train_df, test_df = encode_categoricals(train_df, test_df, cat_cols)
    tr, ho = train_test_split(train_df, test_size=HOLDOUT_FRAC, stratify=train_df["y"], random_state=SPLIT_SEED)
    tr, ho = tr.reset_index(drop=True), ho.reset_index(drop=True)
    if QUICK:
        tr = tr.iloc[strat_sample(tr["y"].values, 16000, SPLIT_SEED)].reset_index(drop=True)
    features = [c for c in tr.columns if c not in ("y", "family") and tr[c].nunique() > 1]
    def xy(d):
        return d[features].to_numpy(dtype=np.float64), d["y"].to_numpy(dtype=int), d["family"].to_numpy(dtype=object)
    Xtr, ytr, ftr = xy(tr)
    Xho, yho, fho = xy(ho)
    Xte, yte, fte = xy(test_df)
    fams = sorted(f for f in pd.unique(ftr) if f not in ("Normal", "Other"))
    log.info("  %-10s features=%d  train=%d (attack %.3f)  held-out=%d  test=%d (attack %.3f)  families=%s",
             name, len(features), len(ytr), ytr.mean(), len(yho), len(yte), yte.mean(),
             ", ".join("%s:%d" % (f, (ftr == f).sum()) for f in fams))
    return {"name": name, "features": features, "Xtr": Xtr, "ytr": ytr, "ftr": ftr, "Xho": Xho, "yho": yho,
            "fho": fho, "Xte": Xte, "yte": yte, "families": fams}


def read_cicids(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    if "Attack Type" not in df.columns:
        sys.exit("CICIDS2017 file has no 'Attack Type' column. Columns found: " + ", ".join(df.columns))
    lab = df["Attack Type"].astype(str).str.strip()
    counts = lab.value_counts()
    log.info("  CICIDS2017 raw file: %d rows, %d columns; label values: %s", len(df), df.shape[1],
             ", ".join("%s:%d" % (k, v) for k, v in counts.items()))
    benign = lab.str.lower().isin(CIC_BENIGN)
    if not benign.any():
        sys.exit("No benign records found in CICIDS2017 (expected one of %s in 'Attack Type')." % (CIC_BENIGN,))
    feats = [c for c in df.columns if c != "Attack Type"]
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = X.isna().any(axis=1)
    if bad.any():
        log.info("  CICIDS2017: dropping %d rows with missing or infinite values", int(bad.sum()))
    X = X[~bad].astype(np.float32)
    X["y"] = (~benign[~bad]).astype(int).to_numpy()
    X["family"] = np.where(X["y"] == 0, "Normal", lab[~bad].to_numpy())
    return X.reset_index(drop=True)


def strat_key(fam, y, min_count=10):
    """Stratification key: the family, or the binary label for very rare families."""
    s = pd.Series(fam)
    vc = s.value_counts()
    rare = s.map(vc) < min_count
    return np.where(rare, np.where(np.asarray(y) == 1, "rare_attack", "rare_normal"), s.to_numpy()).astype(str)


def prepare_cicids(df):
    from sklearn.model_selection import train_test_split
    key = [c for c in df.columns if c != "family"]
    n0 = len(df)
    df = df.drop_duplicates(subset=key).reset_index(drop=True)
    log.info("  CICIDS2017: %d duplicate rows removed, %d remain", n0 - len(df), len(df))
    tr, ho = train_test_split(df, test_size=HOLDOUT_FRAC, stratify=strat_key(df["family"], df["y"]),
                              random_state=SPLIT_SEED)
    del df
    tr, ho = tr.reset_index(drop=True), ho.reset_index(drop=True)
    features = [c for c in tr.columns if c not in ("y", "family") and tr[c].nunique() > 1]
    def xy(d):
        return d[features].to_numpy(dtype=np.float32), d["y"].to_numpy(dtype=int), d["family"].to_numpy(dtype=object)
    Xtr, ytr, ftr = xy(tr)
    Xho, yho, fho = xy(ho)
    train_n = 16000 if QUICK else CIC_TRAIN_N
    holdout_n = 20000 if QUICK else CIC_HOLDOUT_N
    frac = min(1.0, train_n / len(ytr))
    all_fams = sorted(f for f in pd.unique(ftr) if f not in ("Normal", "Other"))
    fams = [f for f in all_fams if (ftr == f).sum() * frac >= (50 if QUICK else MIN_FAMILY)]
    log.info("  %-10s features=%d  train partition=%d (attack %.3f, sample %d per seed)  held-out=%d (sample %d per seed)"
             "  no test file", "CICIDS2017", len(features), len(ytr), ytr.mean(), train_n, len(yho), holdout_n)
    log.info("  CICIDS2017 families in the per-seed training sample (expected): %s",
             ", ".join("%s:%d" % (f, round((ftr == f).sum() * frac)) for f in all_fams))
    log.info("  CICIDS2017 family-noise conditions for: %s (others below %d records)", ", ".join(fams) or "none",
             50 if QUICK else MIN_FAMILY)
    return {"name": "CICIDS2017", "features": features, "Xtr": Xtr, "ytr": ytr, "ftr": ftr, "Xho": Xho, "yho": yho,
            "fho": fho, "Xte": None, "yte": None, "families": fams, "train_n": train_n, "holdout_n": holdout_n}


def load_datasets(papers, names):
    out = []
    p6 = os.path.join(papers, "Paper6")
    if "NSL-KDD" in names:
        nsl_tr = first_existing(os.path.join(p6, "nsl-kdd", "KDDTrain+.txt"), os.path.join(p6, "KDDTrain+.txt"))
        nsl_te = first_existing(os.path.join(p6, "nsl-kdd", "KDDTest+.txt"), os.path.join(p6, "KDDTest+.txt"))
        out.append(prepare("NSL-KDD", read_nsl(nsl_tr), read_nsl(nsl_te), NSL_CAT))
    if "UNSW-NB15" in names:
        out.append(prepare("UNSW-NB15", read_unsw(os.path.join(p6, "UNSW_NB15_training-set.csv")),
                           read_unsw(os.path.join(p6, "UNSW_NB15_testing-set.csv")), UNSW_CAT))
    if "CICIDS2017" in names:
        cic = first_existing(os.path.join(papers, "Paper7", "cicids2017_cleaned.csv"),
                             os.path.join(papers, "Paper9", "cicids2017_cleaned.csv"))
        out.append(prepare_cicids(read_cicids(cic)))
    return out


# ------------------------------------------------------------------ noise conditions
def conditions(families):
    c = [("clean", "clean", 0.0, "")]
    c += [("sym_%d" % int(r * 100), "sym", r, "") for r in (0.05, 0.10, 0.20, 0.30)]
    c += [("a2b_%d" % int(r * 100), "a2b", r, "") for r in (0.05, 0.10, 0.20, 0.40)]
    c += [("b2a_%d" % int(r * 100), "b2a", r, "") for r in (0.05, 0.10, 0.20)]
    c += [("fam_%s" % f, "fam", FAMILY_RATE, f) for f in families]
    return c


def corrupt(y, fam, kind, rate, family, rng):
    y_noisy = y.copy()
    u = rng.random_sample(len(y))
    if kind == "sym":
        flip = u < rate
    elif kind == "a2b":
        flip = (y == 1) & (u < rate)
    elif kind == "b2a":
        flip = (y == 0) & (u < rate)
    elif kind == "fam":
        flip = (fam == family) & (u < rate)
    else:
        flip = np.zeros(len(y), dtype=bool)
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy, flip


# ------------------------------------------------------------------ detectors
def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_lgbm(seed, n_estimators):
    if FALLBACK:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=n_estimators, random_state=seed)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=n_estimators, num_leaves=31, learning_rate=0.1, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1)


def build_and_fit(name, X, y, seed):
    if name == "LR":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0, random_state=seed)).fit(X, y)
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=30 if QUICK else 100, min_samples_leaf=2, max_features="sqrt",
                                      n_jobs=-1, random_state=seed).fit(X, y)
    if name == "XGB":
        if FALLBACK:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_iter=50, max_depth=6, random_state=seed).fit(X, y)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50 if QUICK else 300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", n_jobs=-1,
                             random_state=seed).fit(X, y)
    if name == "LGBM":
        return make_lgbm(seed, 50 if QUICK else 300).fit(X, y)
    if name == "IF":
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=100 if QUICK else 300, max_samples=256, n_jobs=-1,
                               random_state=seed).fit(X[y == 0])      # records labeled benign
    raise ValueError(name)


def scores(name, model, X):
    if name == "IF":
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ cleaner
def confident_flags(X, y, seed):
    """Binary confident-learning rule on 5-fold out-of-fold probabilities."""
    from sklearn.model_selection import StratifiedKFold
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(X, y):
        m = make_lgbm(seed, 50 if QUICK else 100).fit(X[tr_i], y[tr_i])
        p[te_i] = m.predict_proba(X[te_i])[:, 1]
    t_att = p[y == 1].mean() if (y == 1).any() else 1.0
    t_ben = (1 - p[y == 0]).mean() if (y == 0).any() else 1.0
    flag = np.zeros(len(y), dtype=bool)
    flag[(y == 0) & (p >= t_att)] = True
    flag[(y == 1) & ((1 - p) >= t_ben)] = True
    return flag, p


# ------------------------------------------------------------------ metrics
def quantile_threshold(normal_scores, alpha):
    if len(normal_scores) == 0:
        return float("inf")
    return float(np.quantile(normal_scores, 1 - alpha, method="higher"))


def rates(score, y, t):
    alarm = score > t
    neg, pos = (y == 0), (y == 1)
    fpr = alarm[neg].mean() if neg.any() else np.nan
    tpr = alarm[pos].mean() if pos.any() else np.nan
    prec = (alarm & pos).sum() / alarm.sum() if alarm.sum() else np.nan
    f1 = 2 * prec * tpr / (prec + tpr) if prec and tpr and not np.isnan(prec) and (prec + tpr) > 0 else 0.0
    return fpr, tpr, prec, f1




# ======================================================================
# v2.0 additions (IJISA revision, Nov 2026)
# ======================================================================
VERSION = "v2.0"
ALL_DATASETS = ["NSL-KDD", "UNSW-NB15", "CICIDS2017", "CICIoT2023"]
VAL_RATES = (0.05, 0.10, 0.20)                 # attack-to-benign rates in V for the selection experiment
SELECT_MODES = ("random", "low", "soft", "high")
CLEANER_CONDS = ("clean", "sym_10", "sym_30", "a2b_10", "a2b_40", "b2a_20")   # plus every fam_* condition
IOT_BENIGN_FRAC = 0.05       # share of each CICIoT2023 benign file that is read
IOT_ATTACK_FRAC = 0.0005     # share of each CICIoT2023 attack file that is read
IOT_CATEGORY = [             # CICIoT2023 class folder -> attack category (Neto et al., 2023)
    ("ddos", "DDoS"), ("dos", "DoS"), ("mirai", "Mirai"), ("recon", "Recon"),
    ("vulnerabilityscan", "Recon"), ("mitm", "Spoofing"), ("dns_spoofing", "Spoofing"),
    ("dictionarybruteforce", "BruteForce"), ("sqlinjection", "Web"), ("xss", "Web"),
    ("commandinjection", "Web"), ("backdoor_malware", "Web"), ("uploading_attack", "Web"),
    ("browserhijacking", "Web")]
TUNE_GRIDS = {
    "LR": {"logisticregression__C": [0.1, 1.0, 10.0]},
    "RF": {"n_estimators": [100, 300], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]},
    "XGB": {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.2]},
    "LGBM": {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1]},
}
TUNED = {}                    # dataset -> model -> best parameters (filled by tune_all)


def hardware_info():
    info = {}
    try:
        import platform
        info["platform"] = platform.platform()
        info["python"] = platform.python_version()
        info["cpu_count"] = os.cpu_count()
        if os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        if os.path.exists("/proc/meminfo"):
            info["mem_kb"] = int(open("/proc/meminfo").readline().split()[1])
    except Exception as e:
        info["error"] = str(e)
    import importlib
    for m in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "torch", "scipy"):
        try:
            info[m] = importlib.import_module(m).__version__
        except Exception:
            info[m] = "n/a"
    return info


# ------------------------------------------------------------------ data on the VM (vmshare/data)
def read_cicids_chunked(path):
    """Memory-saving reader for the cleaned CICIDS2017 release (same result as read_cicids)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    parts = []
    for ch in pd.read_csv(path, sep=sep, low_memory=False, chunksize=250000):
        ch.columns = [str(c).strip().lstrip("﻿") for c in ch.columns]
        lab = ch["Attack Type"].astype(str).str.strip()
        X = ch.drop(columns=["Attack Type"]).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        bad = X.isna().any(axis=1)
        X = X[~bad].astype(np.float32)
        lab = lab[~bad]
        X["y"] = (~lab.str.lower().isin(CIC_BENIGN)).astype(int).to_numpy()
        X["family"] = np.where(X["y"] == 0, "Normal", lab.to_numpy())
        parts.append(X)
    df = pd.concat(parts, ignore_index=True)
    log.info("  CICIDS2017 file: %d usable rows; label values: %s", len(df),
             ", ".join("%s:%d" % kv for kv in df["family"].value_counts().items()))
    return df


def iot_category(folder):
    f = folder.lower()
    if f.startswith("benign"):
        return "Normal"
    for prefix, cat in IOT_CATEGORY:
        if f.startswith(prefix):
            return cat
    return "Other"


def read_ciciot(zip_path, cache_path):
    """Read CICIoT2023 (CSV.zip from the providers) file by file, keeping a fixed random share of each file.
    Benign files are read at a higher rate than attack files so that the attack share is closer to that
    of the other datasets. The sample is cached, so later runs use exactly the same records."""
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        log.info("  CICIoT2023: cached sample %s (%d rows)", cache_path, len(df))
        return df
    import zipfile
    z = zipfile.ZipFile(zip_path)
    parts, names = [], sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
    for i, n in enumerate(names):
        folder = n.replace("\\", "/").split("/")[-2]
        cat = iot_category(folder)
        frac = IOT_BENIGN_FRAC if cat == "Normal" else IOT_ATTACK_FRAC
        ch = pd.read_csv(z.open(n), low_memory=False)
        k = int(round(frac * len(ch)))
        if k == 0:
            continue
        ch = ch.sample(n=k, random_state=SPLIT_SEED + i)
        ch = ch.drop(columns=[c for c in ch.columns if c.strip().lower() == "label"])
        ch["y"] = int(cat != "Normal")
        ch["family"] = cat
        ch["_folder"] = folder
        parts.append(ch)
        if (i + 1) % 20 == 0:
            log.info("  CICIoT2023: read %d of %d files", i + 1, len(names))
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in ("y", "family", "_folder")]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    df = df.dropna(subset=feats).reset_index(drop=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    log.info("  CICIoT2023: sampled %d rows from %d files (benign %.3f, attack %.4f of each file); classes: %s",
             len(df), len(names), IOT_BENIGN_FRAC, IOT_ATTACK_FRAC,
             ", ".join("%s:%d" % kv for kv in df["family"].value_counts().items()))
    return df


def prepare_holdout_only(name, df):
    """Datasets without a separate test file and without a per-seed cap (CICIoT2023)."""
    from sklearn.model_selection import train_test_split
    df = df.drop(columns=[c for c in ("_folder",) if c in df.columns])
    key = [c for c in df.columns if c != "family"]
    n0 = len(df)
    df = df.drop_duplicates(subset=key).reset_index(drop=True)
    log.info("  %s: %d duplicate rows removed, %d remain", name, n0 - len(df), len(df))
    tr, ho = train_test_split(df, test_size=HOLDOUT_FRAC, stratify=strat_key(df["family"], df["y"]),
                              random_state=SPLIT_SEED)
    tr, ho = tr.reset_index(drop=True), ho.reset_index(drop=True)
    if QUICK:
        tr = tr.iloc[strat_sample(tr["y"].values, 8000, SPLIT_SEED)].reset_index(drop=True)
    features = [c for c in tr.columns if c not in ("y", "family") and tr[c].nunique() > 1]
    def xy(d):
        return d[features].to_numpy(dtype=np.float64), d["y"].to_numpy(dtype=int), d["family"].to_numpy(dtype=object)
    Xtr, ytr, ftr = xy(tr)
    Xho, yho, fho = xy(ho)
    all_fams = sorted(f for f in pd.unique(ftr) if f not in ("Normal", "Other"))
    fams = [f for f in all_fams if (ftr == f).sum() * TRAIN_FRAC >= (50 if QUICK else MIN_FAMILY)]
    log.info("  %-10s features=%d  train=%d (attack %.3f)  held-out=%d  no test file  families=%s  family-noise for: %s",
             name, len(features), len(ytr), ytr.mean(), len(yho),
             ", ".join("%s:%d" % (f, (ftr == f).sum()) for f in all_fams), ", ".join(fams) or "none")
    return {"name": name, "features": features, "Xtr": Xtr, "ytr": ytr, "ftr": ftr, "Xho": Xho, "yho": yho,
            "fho": fho, "Xte": None, "yte": None, "families": fams}


def load_datasets_vm(data_dir, cache_dir, names):
    out = []
    p12 = os.path.join(data_dir, "p12")
    if "NSL-KDD" in names:
        out.append(prepare("NSL-KDD", read_nsl(os.path.join(p12, "KDDTrain+.txt")),
                           read_nsl(os.path.join(p12, "KDDTest+.txt")), NSL_CAT))
    if "UNSW-NB15" in names:
        out.append(prepare("UNSW-NB15", read_unsw(os.path.join(p12, "UNSW_NB15_training-set.csv")),
                           read_unsw(os.path.join(p12, "UNSW_NB15_testing-set.csv")), UNSW_CAT))
    if "CICIDS2017" in names:
        out.append(prepare_cicids(read_cicids_chunked(os.path.join(p12, "cicids2017_cleaned.csv"))))
    if "CICIoT2023" in names:
        df = read_ciciot(os.path.join(data_dir, "iot", "ciciot2023", "CSV.zip"),
                         os.path.join(cache_dir, "ciciot2023_p9_sample.csv.gz"))
        out.append(prepare_holdout_only("CICIoT2023", df))
    return out


def seed_split(ds, seed):
    """Per-seed training sample and V / C split, exactly as in v1.1 (CICIoT2023 follows NSL-KDD and UNSW-NB15)."""
    if "train_n" in ds:
        sub = strat_sample(strat_key(ds["ftr"], ds["ytr"]), ds["train_n"], seed)
        ho = strat_sample(strat_key(ds["fho"], ds["yho"]), ds["holdout_n"], seed)
        ho_key = strat_key(ds["fho"][ho], ds["yho"][ho])
        idx_v = ho[strat_sample(ho_key, len(ho) // 2, seed)]
        idx_c = np.setdiff1d(ho, idx_v)
    else:
        sub = strat_sample(ds["ytr"], int(round(TRAIN_FRAC * len(ds["ytr"]))), seed)
        idx_v = strat_sample(ds["yho"], len(ds["yho"]) // 2, seed)
        idx_c = np.setdiff1d(np.arange(len(ds["yho"])), idx_v)
    return sub, idx_v, idx_c


# ------------------------------------------------------------------ tuned detectors (reviewer comment M5)
def build_tuned(name, X, y, seed, params):
    if name == "LR":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        from sklearn.linear_model import LogisticRegression
        m = make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                          LogisticRegression(max_iter=3000, random_state=seed))
        m.set_params(**params)
        return m.fit(X, y)
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params).fit(X, y)
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50 if QUICK else 300, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             eval_metric="logloss", n_jobs=-1, random_state=seed, **params).fit(X, y)
    if name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=50 if QUICK else 300, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1, **params).fit(X, y)
    raise ValueError(name)


def tune_all(datasets, out_dir, seed=42):
    """Grid search per dataset and detector on the clean training sample of one seed (3-fold stratified CV,
    scoring = standardized partial ROC AUC up to FPR 0.01, the region of the 1% budget). The chosen settings
    are then fixed for all seeds and noise conditions of that dataset."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    path = os.path.join(out_dir, "tuned_params.json")
    if os.path.exists(path):
        TUNED.update(json.load(open(path)))
        log.info("tuned parameters loaded from %s", path)
        return
    scorer = make_scorer(roc_auc_score, max_fpr=0.01, response_method="predict_proba")
    grid_rows = []
    for ds in datasets:
        sub, _, _ = seed_split(ds, seed)
        X, y = ds["Xtr"][sub], ds["ytr"][sub]
        TUNED[ds["name"]] = {}
        for name in SUPERVISED:
            if name == "LR":
                est = make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                                    LogisticRegression(max_iter=3000, random_state=seed))
            elif name == "RF":
                est = RandomForestClassifier(n_jobs=-1, random_state=seed)
            elif name == "XGB":
                from xgboost import XGBClassifier
                est = XGBClassifier(n_estimators=50 if QUICK else 300, subsample=0.8, colsample_bytree=0.8,
                                    tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed)
            else:
                from lightgbm import LGBMClassifier
                est = LGBMClassifier(n_estimators=50 if QUICK else 300, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1)
            t0 = time.time()
            gs = GridSearchCV(est, TUNE_GRIDS[name], scoring=scorer, refit=False, n_jobs=1,
                              cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)).fit(X, y)
            best = gs.best_params_
            TUNED[ds["name"]][name] = best
            for p, s, sd in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"],
                                gs.cv_results_["std_test_score"]):
                grid_rows.append({"dataset": ds["name"], "model": name, "params": json.dumps(p), "pauc_mean": s,
                                  "pauc_sd": sd, "chosen": p == best})
            log.info("  tuned %-10s %-4s best=%s  pAUC=%.4f  (%.0fs)", ds["name"], name, best, gs.best_score_,
                     time.time() - t0)
    json.dump(TUNED, open(path, "w"), indent=1)
    pd.DataFrame(grid_rows).to_csv(os.path.join(out_dir, "tuning_grid.csv"), index=False)


def run_seed_tuned(seed, datasets, seed_dir):
    """All noise conditions, no remedy, the four supervised detectors with tuned settings."""
    from sklearn.metrics import roc_auc_score
    os.makedirs(seed_dir, exist_ok=True)
    rows = []
    for ds in datasets:
        sub, idx_v, idx_c = seed_split(ds, seed)
        X, y, fam = ds["Xtr"][sub], ds["ytr"][sub], ds["ftr"][sub]
        Xv, yv, fv = ds["Xho"][idx_v], ds["yho"][idx_v], ds["fho"][idx_v]
        Xc, yc, fc = ds["Xho"][idx_c], ds["yho"][idx_c], ds["fho"][idx_c]
        for ci, (cname, kind, rate, family) in enumerate(conditions(ds["families"])):
            t0 = time.time()
            y_noisy, flipped = corrupt(y, fam, kind, rate, family, np.random.RandomState(seed * 1000 + ci))
            yv_noisy, _ = corrupt(yv, fv, kind, rate, family, np.random.RandomState(seed * 1000 + ci + 500))
            for m in SUPERVISED:
                model = build_tuned(m, X, y_noisy, seed, TUNED[ds["name"]][m])
                sv, sc = scores(m, model, Xv), scores(m, model, Xc)
                row = {"dataset": ds["name"], "seed": seed, "condition": cname, "kind": kind, "rate": rate,
                       "family": family, "model": m, "auc_control": roc_auc_score(yc, sc)}
                for tname, t in (("val", quantile_threshold(sv[yv == 0], ALPHA)),
                                 ("noisyval", quantile_threshold(sv[yv_noisy == 0], ALPHA))):
                    fpr, tpr, _, _ = rates(sc, yc, t)
                    row["fpr_%s_control" % tname], row["recall_%s_control" % tname] = fpr, tpr
                if kind == "fam":
                    mask = fc == family
                    t = quantile_threshold(sv[yv == 0], ALPHA)
                    row["family_recall_val_control"] = float((sc[mask] > t).mean()) if mask.any() else np.nan
                rows.append(row)
            log.info("[tuned seed %d] %-10s %-13s %.1fs", seed, ds["name"], cname, time.time() - t0)
    pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "results_tuned.csv"), index=False)


# ------------------------------------------------------------------ which attacks are mislabeled in V (M1)
def select_mislabeled(ref_score, attack_idx, rate, mode, rng):
    """Indices (into V) of the attacks that are labeled benign. ref_score orders the attacks by how
    attack-like a reference model finds them (low = hard to recognize)."""
    k = int(round(rate * len(attack_idx)))
    if k == 0:
        return np.array([], dtype=int)
    s = ref_score[attack_idx]
    if mode == "random":
        return rng.choice(attack_idx, k, replace=False)
    if mode == "low":
        return attack_idx[np.argsort(s, kind="stable")[:k]]
    if mode == "high":
        return attack_idx[np.argsort(-s, kind="stable")[:k]]
    if mode == "soft":            # probability falls linearly with the rank: low-scoring attacks are more likely
        r = pd.Series(s).rank(method="first").to_numpy()
        w = (len(s) + 1 - r)
        return rng.choice(attack_idx, k, replace=False, p=w / w.sum())
    raise ValueError(mode)


def run_seed_select(seed, datasets, seed_dir):
    """Attacks in V labeled benign at random (as in Table 4) or depending on how hard they are to recognize.
    The detectors are trained exactly as in the a2b condition of the same rate (same training noise), so the
    'random' mode reproduces the a2b rows of Table 4 and only the choice of the mislabeled V records differs."""
    os.makedirs(seed_dir, exist_ok=True)
    rows = []
    for ds in datasets:
        sub, idx_v, idx_c = seed_split(ds, seed)
        X, y, fam = ds["Xtr"][sub], ds["ytr"][sub], ds["ftr"][sub]
        Xv, yv = ds["Xho"][idx_v], ds["yho"][idx_v]
        Xc, yc = ds["Xho"][idx_c], ds["yho"][idx_c]
        # reference labeler: LightGBM trained on a random half of the clean training sample
        rs = np.random.RandomState(seed + 7000)
        half = rs.rand(len(y)) < 0.5
        ref = make_lgbm(seed + 7000, 50 if QUICK else 100).fit(X[half], y[half])
        ref_v = ref.predict_proba(Xv)[:, 1]
        att = np.flatnonzero(yv == 1)
        conds = [c[0] for c in conditions(ds["families"])]
        for rate in VAL_RATES:
            ci = conds.index("a2b_%d" % int(rate * 100))
            y_noisy, _ = corrupt(y, fam, "a2b", rate, "", np.random.RandomState(seed * 1000 + ci))
            models = {m: build_and_fit(m, X, y_noisy, seed) for m in SUPERVISED}
            sv = {m: scores(m, mod, Xv) for m, mod in models.items()}
            sc = {m: scores(m, mod, Xc) for m, mod in models.items()}
            for mode in SELECT_MODES:
                if mode == "random":       # same random stream as the a2b condition in v1.1
                    yv_noisy, _ = corrupt(yv, None, "a2b", rate, "", np.random.RandomState(seed * 1000 + ci + 500))
                    mis = np.flatnonzero((yv == 1) & (yv_noisy == 0))
                else:
                    mis = select_mislabeled(ref_v, att, rate, mode, np.random.RandomState(seed * 100 + int(rate * 100)))
                    yv_noisy = yv.copy()
                    yv_noisy[mis] = 0
                lab0 = yv_noisy == 0
                c = len(mis) / lab0.sum()
                for m in SUPERVISED:
                    t_clean = quantile_threshold(sv[m][yv == 0], ALPHA)
                    t_noisy = quantile_threshold(sv[m][lab0], ALPHA)
                    fpr, tpr, _, _ = rates(sc[m], yc, t_noisy)
                    _, tpr_clean, _, _ = rates(sc[m], yc, t_clean)
                    q = float((sv[m][mis] > t_clean).mean()) if len(mis) else np.nan
                    rows.append({"dataset": ds["name"], "seed": seed, "rate": rate, "mode": mode, "model": m,
                                 "n_mislabeled": len(mis), "c": c, "ceiling": min(1.0, ALPHA / c) if c > 0 else 1.0,
                                 "recall_clean_v": tpr_clean, "recall_noisy_v": tpr, "fpr_noisy_v": fpr,
                                 "share_mislabeled_above_clean_tau": q,
                                 "ties_at_noisy_tau": float((sv[m][lab0] == t_noisy).mean()),
                                 "ref_score_mislabeled_mean": float(ref_v[mis].mean()) if len(mis) else np.nan,
                                 "ref_score_attacks_mean": float(ref_v[att].mean())})
        log.info("[select seed %d] %-10s done", seed, ds["name"])
    pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "results_select.csv"), index=False)


# ------------------------------------------------------------------ further cleaners (M2)
def ensemble_flags(X, y, seed):
    """Classification filter of Brodley and Friedl with LR, RF and LightGBM on 5-fold out-of-fold predictions.
    Returns the majority and the consensus flags."""
    from sklearn.model_selection import StratifiedKFold
    wrong = np.zeros((3, len(y)), dtype=bool)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(X, y):
        for j, name in enumerate(("LR", "RF", "LGBM")):
            if name == "LGBM":
                m = make_lgbm(seed, 50 if QUICK else 100).fit(X[tr_i], y[tr_i])
            else:
                m = build_and_fit(name, X[tr_i], y[tr_i], seed)
            wrong[j, te_i] = (m.predict_proba(X[te_i])[:, 1] >= 0.5).astype(int) != y[te_i]
    votes = wrong.sum(axis=0)
    return votes >= 2, votes == 3


def coteaching_flags(X, y, seed, forget_rate, epochs=None):
    """Co-teaching (Han et al., 2018) with two small MLPs. Each network trains on the small-loss records chosen
    by its peer; after training, the records with the largest mean loss (share = forget_rate) are flagged."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    epochs = epochs or (4 if QUICK else 15)
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(signed_log(X)).astype(np.float32)
    Xt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.int64))
    def net():
        return nn.Sequential(nn.Linear(Z.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    n1, n2 = net(), net()
    o1 = torch.optim.Adam(n1.parameters(), lr=1e-3)
    o2 = torch.optim.Adam(n2.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(reduction="none")
    g = torch.Generator().manual_seed(seed)
    ramp = max(1, epochs // 3)
    for ep in range(epochs):
        fr = forget_rate * min(1.0, (ep + 1) / ramp)
        perm = torch.randperm(len(yt), generator=g)
        for b in range(0, len(yt), 256):
            i = perm[b:b + 256]
            l1, l2 = ce(n1(Xt[i]), yt[i]), ce(n2(Xt[i]), yt[i])
            keep = max(1, int(round((1 - fr) * len(i))))
            k1 = torch.argsort(l1.detach())[:keep]       # records net 1 finds clean -> train net 2
            k2 = torch.argsort(l2.detach())[:keep]
            o1.zero_grad(); ce(n1(Xt[i][k2]), yt[i][k2]).mean().backward(); o1.step()
            o2.zero_grad(); ce(n2(Xt[i][k1]), yt[i][k1]).mean().backward(); o2.step()
    with torch.no_grad():
        loss = (ce(n1(Xt), yt) + ce(n2(Xt), yt)).numpy() / 2
    k = int(round(forget_rate * len(y)))
    flag = np.zeros(len(y), dtype=bool)
    if k > 0:
        flag[np.argsort(-loss)[:k]] = True
    return flag


def run_seed_cleaners(seed, datasets, seed_dir):
    """Confident learning, the ensemble filter (majority, consensus) and co-teaching; remedy = removal."""
    os.makedirs(seed_dir, exist_ok=True)
    rows, crow = [], []
    for ds in datasets:
        sub, idx_v, idx_c = seed_split(ds, seed)
        X, y, fam = ds["Xtr"][sub], ds["ytr"][sub], ds["ftr"][sub]
        Xv, yv = ds["Xho"][idx_v], ds["yho"][idx_v]
        Xc, yc, fc = ds["Xho"][idx_c], ds["yho"][idx_c], ds["fho"][idx_c]
        for ci, (cname, kind, rate, family) in enumerate(conditions(ds["families"])):
            if cname not in CLEANER_CONDS and kind != "fam":
                continue
            t0 = time.time()
            y_noisy, flipped = corrupt(y, fam, kind, rate, family, np.random.RandomState(seed * 1000 + ci))
            cl, _ = confident_flags(X, y_noisy, seed)
            maj, cons = ensemble_flags(X, y_noisy, seed)
            est = float(cl.mean())                     # noise rate estimated by confident learning
            cot = coteaching_flags(X, y_noisy, seed, forget_rate=min(0.5, est))
            cot_o = coteaching_flags(X, y_noisy, seed, forget_rate=min(0.5, float(flipped.mean())))   # oracle rate
            for cleaner, flag in (("none", np.zeros(len(y), bool)), ("confident", cl), ("ens_majority", maj),
                                  ("ens_consensus", cons), ("coteaching", cot), ("coteaching_oracle", cot_o)):
                crow.append({"dataset": ds["name"], "seed": seed, "condition": cname, "kind": kind, "rate": rate,
                             "family": family, "cleaner": cleaner, "n_train": len(y), "n_flipped": int(flipped.sum()),
                             "n_flagged": int(flag.sum()), "flagged_flipped": int((flag & flipped).sum()),
                             "flagged_correct": int((flag & ~flipped).sum()), "forget_rate_est": min(0.5, est),
                             "forget_rate_true": min(0.5, float(flipped.mean()))})
                keep = ~flag
                for m in MODELS:
                    yk = y_noisy[keep]
                    if m in SUPERVISED and len(np.unique(yk)) < 2:
                        continue
                    model = build_and_fit(m, X[keep], yk, seed)
                    sv, sc = scores(m, model, Xv), scores(m, model, Xc)
                    t = quantile_threshold(sv[yv == 0], ALPHA)
                    fpr, tpr, _, _ = rates(sc, yc, t)
                    row = {"dataset": ds["name"], "seed": seed, "condition": cname, "kind": kind, "rate": rate,
                           "family": family, "cleaner": cleaner, "model": m, "recall_val_control": tpr,
                           "fpr_val_control": fpr, "n_train_used": int(keep.sum())}
                    if kind == "fam":
                        mask = fc == family
                        row["family_recall_val_control"] = float((sc[mask] > t).mean()) if mask.any() else np.nan
                    rows.append(row)
            log.info("[cleaners seed %d] %-10s %-13s flipped=%d  CL=%d/%d  maj=%d/%d  cons=%d/%d  cot=%d/%d  %.0fs",
                     seed, ds["name"], cname, flipped.sum(), (cl & flipped).sum(), cl.sum(), (maj & flipped).sum(),
                     maj.sum(), (cons & flipped).sum(), cons.sum(), (cot & flipped).sum(), cot.sum(), time.time() - t0)
    pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "results_cleaners.csv"), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(seed_dir, "cleaners.csv"), index=False)


# ------------------------------------------------------------------ the v1.1 study on CICIoT2023 (M4)
def run_seed_full(seed, datasets, seed_dir):
    """Same conditions, remedies and detectors as v1.1 (used for the new dataset)."""
    from sklearn.metrics import roc_auc_score
    os.makedirs(seed_dir, exist_ok=True)
    rows, clean_rows = [], []
    for ds in datasets:
        sub, idx_v, idx_c = seed_split(ds, seed)
        X, y, fam = ds["Xtr"][sub], ds["ytr"][sub], ds["ftr"][sub]
        Xv, yv, fv = ds["Xho"][idx_v], ds["yho"][idx_v], ds["fho"][idx_v]
        Xc, yc, fc = ds["Xho"][idx_c], ds["yho"][idx_c], ds["fho"][idx_c]
        for ci, (cname, kind, rate, family) in enumerate(conditions(ds["families"])):
            t0 = time.time()
            rng = np.random.RandomState(seed * 1000 + ci)
            y_noisy, flipped = corrupt(y, fam, kind, rate, family, rng)
            yv_noisy, _ = corrupt(yv, fv, kind, rate, family, np.random.RandomState(seed * 1000 + ci + 500))
            flag, _ = confident_flags(X, y_noisy, seed)
            clean_rows.append({"dataset": ds["name"], "seed": seed, "condition": cname, "kind": kind, "rate": rate,
                               "family": family, "n_train": len(y), "n_flipped": int(flipped.sum()),
                               "n_flagged": int(flag.sum()), "flagged_flipped": int((flag & flipped).sum()),
                               "flagged_correct_attacks": int((flag & ~flipped & (y == 1)).sum()),
                               "flagged_correct_benign": int((flag & ~flipped & (y == 0)).sum()),
                               "noisy_label_rate": float(flipped.mean())})
            for remedy in REMEDIES:
                if remedy == "none":
                    Xr, yr = X, y_noisy
                elif remedy == "remove":
                    Xr, yr = X[~flag], y_noisy[~flag]
                else:
                    Xr, yr = X, np.where(flag, 1 - y_noisy, y_noisy)
                for m in MODELS:
                    if m == "IF" and remedy == "relabel":
                        continue
                    if m in SUPERVISED and len(np.unique(yr)) < 2:
                        continue
                    tf = time.time()
                    model = build_and_fit(m, Xr, yr, seed)
                    fit_s = time.time() - tf
                    sv, sc = scores(m, model, Xv), scores(m, model, Xc)
                    t_clean = quantile_threshold(sv[yv == 0], ALPHA)
                    t_noisy = quantile_threshold(sv[yv_noisy == 0], ALPHA)
                    row = {"dataset": ds["name"], "seed": seed, "condition": cname, "kind": kind, "rate": rate,
                           "family": family, "remedy": remedy, "model": m, "fit_seconds": round(fit_s, 2),
                           "n_train_used": len(yr), "auc_control": roc_auc_score(yc, sc)}
                    for tname, t in (("val", t_clean), ("noisyval", t_noisy)):
                        fpr, tpr, _, _ = rates(sc, yc, t)
                        row["fpr_%s_control" % tname], row["recall_%s_control" % tname] = fpr, tpr
                    if m in SUPERVISED:
                        fpr, tpr, prec, f1 = rates(sc, yc, 0.5)
                        row.update({"fpr_05_control": fpr, "recall_05_control": tpr, "precision_05_control": prec,
                                    "f1_05_control": f1})
                    if kind == "fam":
                        mask = fc == family
                        if mask.any():
                            row["family_recall_val_control"] = float((sc[mask] > t_clean).mean())
                    rows.append(row)
            log.info("[full seed %d] %-10s %-13s flipped=%5d flagged=%5d (of which flipped %5d)  %.1fs",
                     seed, ds["name"], cname, flipped.sum(), flag.sum(), (flag & flipped).sum(), time.time() - t0)
    pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "results.csv"), index=False)
    pd.DataFrame(clean_rows).to_csv(os.path.join(seed_dir, "cleaner.csv"), index=False)


PARTS = {"select": run_seed_select, "tuned": run_seed_tuned, "cleaners": run_seed_cleaners, "full": run_seed_full}


def main():
    global QUICK
    ap = argparse.ArgumentParser(description="Paper 9 v2.0 runs for the IJISA revision")
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--data-dir", required=True, help="folder with p12/ and iot/ (vmshare/data)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/P9_v2_cache"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    QUICK = args.quick
    globals()["QUICK"] = QUICK
    out = os.path.join(args.out_dir + ("_quicktest" if QUICK else ""), args.part)
    os.makedirs(out, exist_ok=True)
    setup_logging(out)
    log.info("=" * 70)
    log.info("Paper 9 %s part=%s datasets=%s seeds=%s quick=%s", VERSION, args.part, args.datasets, args.seeds, QUICK)
    hw = hardware_info()
    log.info("hardware/software: %s", json.dumps(hw))
    json.dump(hw, open(os.path.join(out, "hardware.json"), "w"), indent=1)
    t_all = time.time()
    datasets = load_datasets_vm(args.data_dir, args.cache_dir, args.datasets)
    if args.part == "tuned":
        tune_all(datasets, out)
    for seed in args.seeds:
        sd = os.path.join(out, "seed_%d" % seed)
        if os.path.exists(os.path.join(sd, "DONE.json")):
            log.info("[seed %d] already finished, skipped", seed)
            continue
        t0 = time.time()
        PARTS[args.part](seed, datasets, sd)
        json.dump({"seed": seed, "minutes": round((time.time() - t0) / 60, 1), "datasets": args.datasets},
                  open(os.path.join(sd, "DONE.json"), "w"))
        log.info("[seed %d] finished in %.1f min", seed, (time.time() - t0) / 60)
    log.info("PART %s DONE in %.1f min", args.part, (time.time() - t_all) / 60)


if __name__ == "__main__":
    main()
