"""
Paper 9 (v1.0) - Mislabeled training data in network intrusion detection:
how label noise of different kinds changes recall at a fixed false-alarm budget, and whether
simple confident-learning-style cleaning helps tree ensembles.

Design (5 Oct 2026)
  * Binary task: normal (0) vs attack (1). Datasets: NSL-KDD and UNSW-NB15 (files in Paper6).
    Duplicates removed; fixed stratified 80/20 split of the official training file into a training
    partition and a held-out partition; the official test file is the shifted traffic T.
  * Per seed (10 seeds, 42-51): a stratified 50% subsample of the training partition is used for
    training; the held-out partition is split at random into a validation half V (thresholds) and a
    clean control half C (never trained on, labels never corrupted).
  * Label-noise conditions applied to the TRAINING labels (and, for the noisy-threshold analysis, the
    same process applied independently to the labels of V):
        clean                    no noise
        sym_r     r = 5,10,20,30%  every label flipped with probability r
        a2b_r     r = 5,10,20,40%  attacks labeled benign with probability r (missed attacks)
        b2a_r     r = 5,10,20%     benign records labeled attack with probability r
        fam_<f>                    50% of the records of one attack family labeled benign
                                   (NSL-KDD: DoS, Probe, R2L, U2R; UNSW-NB15: the 9 categories)
  * Remedies: none | remove (drop records flagged by the cleaner) | relabel (flip flagged labels).
    Cleaner: binary confident-learning rule (Northcutt et al., JAIR 2021) on 5-fold out-of-fold
    probabilities from a LightGBM model: a record labeled benign is flagged if p(attack) >= mean
    p(attack) of records labeled attack; a record labeled attack is flagged if p(benign) >= mean
    p(benign) of records labeled benign.
  * Detectors: Logistic Regression, Random Forest, XGBoost, LightGBM (supervised) and Isolation Forest
    trained on the records labeled benign (so missed attacks contaminate it). IF uses none/remove.
  * Metrics on C and T: ROC AUC; threshold at a 1% target FPR from the normal records of V (clean
    labels) -> realized FPR and recall; same with the threshold from the noisy labels of V; for
    supervised detectors also FPR, recall, precision and F1 at the 0.5 threshold. On C, recall per
    attack family. Cleaner statistics: flagged, flagged that were really flipped, flipped found,
    correctly labeled attacks discarded.

How to run (Anaconda Prompt, same environment as P12/P13)
  conda activate p12
  cd /d "H:\\My Drive\\Publish\\Papers\\Paper9"
  python P9_run_v1.0.py --quick        (check everything works, a few minutes)
  python P9_run_v1.0.py                (full run)

Other options
  --seeds 42 43           run only these seeds
  --aggregate-only        rebuild summary tables and figures from finished seeds
  --papers-dir PATH       folder that contains Paper6 and Paper9 (default: H:, then G:, then Colab path)

Outputs: Paper9/results_v1.0/ (Paper9/results_v1.0_quicktest/ with --quick); log in run_log.txt.
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback
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
    idx, _ = train_test_split(pool, train_size=n, stratify=y[pool], random_state=seed)
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


def load_datasets(papers):
    p6 = os.path.join(papers, "Paper6")
    nsl_tr = first_existing(os.path.join(p6, "nsl-kdd", "KDDTrain+.txt"), os.path.join(p6, "KDDTrain+.txt"))
    nsl_te = first_existing(os.path.join(p6, "nsl-kdd", "KDDTest+.txt"), os.path.join(p6, "KDDTest+.txt"))
    return [prepare("NSL-KDD", read_nsl(nsl_tr), read_nsl(nsl_te), NSL_CAT),
            prepare("UNSW-NB15", read_unsw(os.path.join(p6, "UNSW_NB15_training-set.csv")),
                    read_unsw(os.path.join(p6, "UNSW_NB15_testing-set.csv")), UNSW_CAT)]


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


def run_seed(seed, datasets, seed_dir):
    from sklearn.metrics import roc_auc_score
    os.makedirs(seed_dir, exist_ok=True)
    rows, clean_rows = [], []
    for ds in datasets:
        sub = strat_sample(ds["ytr"], int(round(TRAIN_FRAC * len(ds["ytr"]))), seed)
        X, y, fam = ds["Xtr"][sub], ds["ytr"][sub], ds["ftr"][sub]
        idx_v = strat_sample(ds["yho"], len(ds["yho"]) // 2, seed)
        idx_c = np.setdiff1d(np.arange(len(ds["yho"])), idx_v)
        Xv, yv, fv = ds["Xho"][idx_v], ds["yho"][idx_v], ds["fho"][idx_v]
        Xc, yc, fc = ds["Xho"][idx_c], ds["yho"][idx_c], ds["fho"][idx_c]
        Xt, yt = ds["Xte"], ds["yte"]
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
                    keep = ~flag
                    Xr, yr = X[keep], y_noisy[keep]
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
                    sv, sc, st = scores(m, model, Xv), scores(m, model, Xc), scores(m, model, Xt)
                    t_clean = quantile_threshold(sv[yv == 0], ALPHA)
                    t_noisy = quantile_threshold(sv[yv_noisy == 0], ALPHA)
                    row = {"dataset": ds["name"], "seed": seed, "condition": cname, "kind": kind, "rate": rate,
                           "family": family, "remedy": remedy, "model": m, "fit_seconds": round(fit_s, 2),
                           "n_train_used": len(yr)}
                    for ev, s_, y_ in (("control", sc, yc), ("test", st, yt)):
                        row["auc_" + ev] = roc_auc_score(y_, s_)
                        for tname, t in (("val", t_clean), ("noisyval", t_noisy)):
                            fpr, tpr, prec, f1 = rates(s_, y_, t)
                            row["fpr_%s_%s" % (tname, ev)] = fpr
                            row["recall_%s_%s" % (tname, ev)] = tpr
                        if m in SUPERVISED:
                            fpr, tpr, prec, f1 = rates(s_, y_, 0.5)
                            row["fpr_05_" + ev] = fpr
                            row["recall_05_" + ev] = tpr
                            row["precision_05_" + ev] = prec
                            row["f1_05_" + ev] = f1
                    if kind == "fam":       # recall on the corrupted family in the clean control set
                        mask = (fc == family)
                        if mask.any():
                            row["family_recall_val_control"] = float((sc[mask] > t_clean).mean())
                    rows.append(row)
            log.info("[seed %d] %-10s %-13s flipped=%5d flagged=%5d (of which flipped %5d)  %.1fs",
                     seed, ds["name"], cname, flipped.sum(), flag.sum(), (flag & flipped).sum(), time.time() - t0)
    pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "results.csv"), index=False)
    pd.DataFrame(clean_rows).to_csv(os.path.join(seed_dir, "cleaner.csv"), index=False)


# ------------------------------------------------------------------ statistics and aggregation
def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - r) * p[i]))
        out[i] = run
    return out


def wilcoxon_p(x, y):
    from scipy.stats import wilcoxon
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def ms(v, fmt="%.3f"):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return "n/a"
    return (fmt + " ± " + fmt) % (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)


def aggregate(out_dir):
    dirs = sorted(d for d in glob.glob(os.path.join(out_dir, "seed_*")) if os.path.exists(os.path.join(d, "DONE.json")))
    if not dirs:
        log.info("No finished seeds yet")
        return
    agg = os.path.join(out_dir, "summary")
    os.makedirs(agg, exist_ok=True)
    R = pd.concat([pd.read_csv(os.path.join(d, "results.csv")) for d in dirs], ignore_index=True)
    CL = pd.concat([pd.read_csv(os.path.join(d, "cleaner.csv")) for d in dirs], ignore_index=True)
    R["family"] = R["family"].fillna("")
    CL["family"] = CL["family"].fillna("")
    seeds = sorted(R.seed.unique())
    datasets = list(dict.fromkeys(R.dataset))
    conds = list(dict.fromkeys(R.condition))

    # (1) effect of noise, remedy = none
    noise_rows, stats = [], []
    for ds in datasets:
        for m in MODELS:
            base = R[(R.dataset == ds) & (R.model == m) & (R.condition == "clean") & (R.remedy == "none")].sort_values("seed")
            for c in conds:
                g = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy == "none")].sort_values("seed")
                if g.empty:
                    continue
                r = {"dataset": ds, "model": m, "condition": c,
                     "recall@1%FPR control": ms(g.recall_val_control), "FPR control": ms(g.fpr_val_control, "%.4f"),
                     "recall@1%FPR test": ms(g.recall_val_test), "FPR test": ms(g.fpr_val_test, "%.4f"),
                     "AUC control": ms(g.auc_control), "AUC test": ms(g.auc_test)}
                if m in SUPERVISED:
                    r.update({"F1@0.5 control": ms(g.f1_05_control), "FPR@0.5 control": ms(g.fpr_05_control, "%.4f"),
                              "recall@0.5 control": ms(g.recall_05_control)})
                if "family_recall_val_control" in g:
                    fr = g.family_recall_val_control.dropna()
                    if len(fr):
                        r["family recall control"] = ms(fr)
                noise_rows.append(r)
                if c != "clean" and len(g) == len(base):
                    for metric in ("recall_val_control", "recall_val_test"):
                        stats.append({"family": "noise vs clean: " + metric, "dataset": ds, "model": m, "condition": c,
                                      "median diff": float(np.median(g[metric].to_numpy() - base[metric].to_numpy())),
                                      "p": wilcoxon_p(g[metric].to_numpy(), base[metric].to_numpy())})
    pd.DataFrame(noise_rows).to_csv(os.path.join(agg, "table_noise_effect.csv"), index=False)

    # family recall for family-noise conditions vs clean (same family)
    fam_rows = []
    for ds in datasets:
        for m in MODELS:
            for c in [c for c in conds if c.startswith("fam_")]:
                g = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy.isin(REMEDIES))]
                r = {"dataset": ds, "model": m, "family": c[4:]}
                for remedy in REMEDIES:
                    h = g[g.remedy == remedy]
                    if not h.empty and "family_recall_val_control" in h:
                        r["family recall (%s)" % remedy] = ms(h.family_recall_val_control)
                fam_rows.append(r)
    pd.DataFrame(fam_rows).to_csv(os.path.join(agg, "table_family_recall.csv"), index=False)

    # (2) remedies
    rem_rows = []
    for ds in datasets:
        for m in MODELS:
            for c in conds:
                none = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy == "none")].sort_values("seed")
                r = {"dataset": ds, "model": m, "condition": c, "none": ms(none.recall_val_control)}
                for remedy in ("remove", "relabel"):
                    g = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy == remedy)].sort_values("seed")
                    if g.empty:
                        continue
                    r[remedy] = ms(g.recall_val_control)
                    r[remedy + " FPR@0.5 control"] = ms(g.fpr_05_control, "%.4f") if m in SUPERVISED else ""
                    if len(g) == len(none):
                        stats.append({"family": "%s vs none: recall_val_control" % remedy, "dataset": ds, "model": m,
                                      "condition": c,
                                      "median diff": float(np.median(g.recall_val_control.to_numpy() - none.recall_val_control.to_numpy())),
                                      "p": wilcoxon_p(g.recall_val_control.to_numpy(), none.recall_val_control.to_numpy())})
                rem_rows.append(r)
    pd.DataFrame(rem_rows).to_csv(os.path.join(agg, "table_remedies.csv"), index=False)

    # (3) thresholds from noisy validation labels (remedy none)
    th_rows = []
    for ds in datasets:
        for m in MODELS:
            for c in conds:
                g = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy == "none")]
                if g.empty:
                    continue
                th_rows.append({"dataset": ds, "model": m, "condition": c,
                                "FPR control (clean V)": ms(g.fpr_val_control, "%.4f"),
                                "FPR control (noisy V)": ms(g.fpr_noisyval_control, "%.4f"),
                                "recall control (clean V)": ms(g.recall_val_control),
                                "recall control (noisy V)": ms(g.recall_noisyval_control)})
    pd.DataFrame(th_rows).to_csv(os.path.join(agg, "table_noisy_validation.csv"), index=False)

    # (4) cleaner statistics
    cl_rows = []
    for ds in datasets:
        for c in conds:
            g = CL[(CL.dataset == ds) & (CL.condition == c)]
            if g.empty:
                continue
            prec = g.flagged_flipped / g.n_flagged.replace(0, np.nan)
            rec = g.flagged_flipped / g.n_flipped.replace(0, np.nan)
            cl_rows.append({"dataset": ds, "condition": c, "flipped": ms(g.n_flipped, "%.0f"),
                            "flagged": ms(g.n_flagged, "%.0f"), "cleaner precision": ms(prec), "cleaner recall": ms(rec),
                            "correct attacks discarded": ms(g.flagged_correct_attacks, "%.0f"),
                            "correct benign discarded": ms(g.flagged_correct_benign, "%.0f")})
    pd.DataFrame(cl_rows).to_csv(os.path.join(agg, "table_cleaner.csv"), index=False)

    st = pd.DataFrame(stats)
    if not st.empty:
        st["p_holm"] = np.nan
        for _, g in st.groupby(["family", "dataset"]):
            st.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
        st["significant (Holm, 0.05)"] = st["p_holm"] < 0.05
        st.round(5).to_csv(os.path.join(agg, "stats_tests.csv"), index=False)
    R.groupby(["dataset", "model"]).fit_seconds.mean().round(2).reset_index().to_csv(os.path.join(agg, "table_timing.csv"), index=False)
    try:
        make_figures(R, datasets, agg)
    except Exception:
        log.error("Figures failed (tables are saved)\n%s", traceback.format_exc())
    with open(os.path.join(agg, "P9_v1.0_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("Paper 9 v1.0 summary (%d seeds: %s)\n\n" % (len(seeds), ", ".join(map(str, seeds))))
        for title, f in [("Effect of label noise (remedy none)", "table_noise_effect.csv"),
                         ("Recall on the corrupted family (control set)", "table_family_recall.csv"),
                         ("Remedies: recall at 1% FPR on control", "table_remedies.csv"),
                         ("Thresholds from noisy validation labels", "table_noisy_validation.csv"),
                         ("Cleaner", "table_cleaner.csv"), ("Tests", "stats_tests.csv")]:
            p = os.path.join(agg, f)
            if os.path.exists(p):
                fh.write("=" * 100 + "\n" + title + "\n" + "=" * 100 + "\n" + pd.read_csv(p).to_string(index=False) + "\n\n")
    log.info("Summary updated from %d seeds: %s", len(seeds), agg)


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color="#e4e2dd", linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(R, datasets, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    kinds = [("sym", "Symmetric flips", [0, 5, 10, 20, 30]), ("a2b", "Attacks labeled benign", [0, 5, 10, 20, 40]),
             ("b2a", "Benign labeled attack", [0, 5, 10, 20])]
    # Fig 1: recall at 1% FPR (control) vs noise rate, remedy none
    fig, axes = plt.subplots(len(datasets), 3, figsize=(10, 3.0 * len(datasets)), sharey="row")
    axes = np.atleast_2d(axes)
    for i, ds in enumerate(datasets):
        for j, (k, title, rs) in enumerate(kinds):
            ax = axes[i, j]
            for m in MODELS:
                ys = []
                for r in rs:
                    c = "clean" if r == 0 else "%s_%d" % (k, r)
                    g = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy == "none")]
                    ys.append(g.recall_val_control.mean() if not g.empty else np.nan)
                ax.plot(rs, ys, marker="o", ms=3.5, lw=1.4, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
            ax.set_title("%s - %s" % (ds, title), fontsize=8.5, color=INK)
            ax.set_xlabel("Noise rate (%)")
            style_axes(ax)
        axes[i, 0].set_ylabel("Recall at 1% FPR (control)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(agg, "fig1_noise_effect.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Fig 2: remedies for a2b noise, RF and XGB
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        for m, ls in (("RF", "-"), ("XGB", "--")):
            for remedy in REMEDIES:
                ys = []
                for r in [0, 5, 10, 20, 40]:
                    c = "clean" if r == 0 else "a2b_%d" % r
                    g = R[(R.dataset == ds) & (R.model == m) & (R.condition == c) & (R.remedy == remedy)]
                    ys.append(g.recall_val_control.mean() if not g.empty else np.nan)
                ax.plot([0, 5, 10, 20, 40], ys, ls=ls, marker="o", ms=3, lw=1.3, color=REMEDY_COLORS[remedy],
                        label="%s, %s" % (m, remedy))
        ax.set_title(ds, fontsize=9, color=INK)
        ax.set_xlabel("Attacks labeled benign (%)")
        style_axes(ax)
    axes[0].set_ylabel("Recall at 1% FPR (control)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(agg, "fig2_remedies.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    global QUICK, FALLBACK
    ap = argparse.ArgumentParser(description="Paper 9 v1.0 run (label noise)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--papers-dir")
    ap.add_argument("--fallback-models", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    QUICK = args.quick
    FALLBACK = args.fallback_models
    papers = find_papers_dir(args.papers_dir)
    out_dir = os.path.join(papers, "Paper9", "results_v1.0_quicktest" if QUICK else "results_v1.0")
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir)
    if args.aggregate_only:
        aggregate(out_dir)
        return
    needed = ("sklearn", "scipy", "matplotlib") if FALLBACK else ("xgboost", "lightgbm", "sklearn", "scipy", "matplotlib")
    missing = []
    for mod in needed:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit("Missing packages: %s. Run:  pip install xgboost lightgbm scikit-learn scipy matplotlib" % ", ".join(missing))
    seeds = args.seeds or (SEEDS[:2] if QUICK else SEEDS)
    awake = keep_awake(True)
    log.info("=" * 70)
    log.info("Paper 9 v1.0 run started | quick=%s | seeds=%s | sleep blocked: %s%s", QUICK, seeds, awake,
             " | FALLBACK MODELS (test only)" if FALLBACK else "")
    datasets = load_datasets(papers)
    t_all = time.time()
    try:
        for seed in seeds:
            sd = os.path.join(out_dir, "seed_%d" % seed)
            if os.path.exists(os.path.join(sd, "DONE.json")):
                log.info("[seed %d] already finished - skipping", seed)
                continue
            t0 = time.time()
            try:
                run_seed(seed, datasets, sd)
                with open(os.path.join(sd, "DONE.json"), "w") as fh:
                    json.dump({"seed": seed, "minutes": round((time.time() - t0) / 60, 1),
                               "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": QUICK}, fh)
                log.info("[seed %d] finished in %.1f min", seed, (time.time() - t0) / 60)
            except Exception:
                log.error("[seed %d] FAILED\n%s", seed, traceback.format_exc())
                continue
            try:
                aggregate(out_dir)
            except Exception:
                log.error("Summary update failed\n%s", traceback.format_exc())
    finally:
        keep_awake(False)
    log.info("All done in %.1f min. Summary: %s", (time.time() - t_all) / 60, os.path.join(out_dir, "summary"))


if __name__ == "__main__":
    main()
