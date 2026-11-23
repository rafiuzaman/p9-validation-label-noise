"""P9 v2.0 analysis for the IJISA revision (23 Nov 2026). Reads v1.0/v1.1 results (NSL-KDD, UNSW-NB15, CICIDS2017)
and the v2.0 parts (select, full = CICIoT2023, tuned, cleaners). Writes summary tables to ana/."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon

import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
R2 = ROOT + "/results/v2.0/"
OUT = ROOT + "/analysis/"
os.makedirs(OUT, exist_ok=True)
rd = lambda p: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(p))], ignore_index=True)
SUP = ["LR", "RF", "XGB", "LGBM"]
PI = {"NSL-KDD": 0.465, "UNSW-NB15": 0.488, "CICIDS2017": 0.170, "CICIoT2023": 0.287}


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1, run)
    return adj


def wx(d):
    d = np.asarray(d, float)
    return 1.0 if np.allclose(d, 0) else wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue


def ms(x):
    return "%.3f ± %.3f" % (np.mean(x), np.std(x, ddof=1))


base = pd.concat([rd(ROOT + "/results/v1.0/seed_*/results.csv"), rd(ROOT + "/results/v1.1/seed_*/results.csv"),
                  rd(R2 + "full/seed_*/results.csv")], ignore_index=True)

# ---------------------------------------------------------------- M4: CICIoT2023 full study
io = base[base.dataset == "CICIoT2023"]
none = io[io.remedy == "none"]
order = ["clean", "sym_5", "sym_10", "sym_20", "sym_30", "a2b_5", "a2b_10", "a2b_20", "a2b_40", "b2a_5", "b2a_10", "b2a_20",
         "fam_DDoS", "fam_DoS", "fam_Mirai"]
t3 = none.pivot_table(index="condition", columns="model", values="recall_val_control", aggfunc=ms).reindex(order)
t3.to_csv(OUT + "M4_ciciot_training_noise_recall.csv")
auc = none.pivot_table(index="condition", columns="model", values="auc_control", aggfunc=ms).reindex(order)
auc.to_csv(OUT + "M4_ciciot_auc.csv")
# noise vs clean tests per model (Holm within model, as in v1.1)
rows = []
for m in ["LR", "RF", "XGB", "LGBM", "IF"]:
    cl = none[(none.model == m) & (none.condition == "clean")].set_index("seed").recall_val_control
    ps, recs = [], []
    for c in order[1:]:
        x = none[(none.model == m) & (none.condition == c)].set_index("seed").recall_val_control
        d = (x - cl).dropna()
        ps.append(wx(d)); recs.append((m, c, d.median(), d.mean()))
    for (m_, c, md, mn), p, pa in zip(recs, ps, holm(ps)):
        rows.append({"model": m_, "condition": c, "median_diff": md, "mean_diff": mn, "p": p, "p_holm": pa, "sig": pa < 0.05})
pd.DataFrame(rows).to_csv(OUT + "M4_ciciot_noise_vs_clean_tests.csv", index=False)
# remedies
rem = io.pivot_table(index=["condition", "model"], columns="remedy", values="recall_val_control", aggfunc="mean").round(3)
rem.to_csv(OUT + "M4_ciciot_remedies.csv")
# Table 4 rows for CICIoT2023
def contam(kind, r, pi):
    return r * pi / (r * pi + 1 - pi) if kind == "a2b" else (r * pi / (r * pi + (1 - r) * (1 - pi)) if kind == "sym" else 0.0)
sup = none[none.model.isin(SUP)]
t4 = []
for c in ["sym_5", "sym_10", "a2b_5", "a2b_10", "a2b_20", "b2a_10"]:
    x = sup[sup.condition == c]; k, r = x.kind.iloc[0], x.rate.iloc[0]; cc = contam(k, r, PI["CICIoT2023"])
    t4.append({"condition": c, "c": round(cc, 3), "ceiling": round(min(1, 0.01 / cc), 2) if cc else 1,
               "recall_clean_V": round(x.recall_val_control.mean(), 3), "recall_noisy_V": round(x.recall_noisyval_control.mean(), 3),
               "fpr_noisy_V_pct": round(100 * x.fpr_noisyval_control.mean(), 2)})
pd.DataFrame(t4).to_csv(OUT + "M4_ciciot_table4_rows.csv", index=False)
fam = none[none.kind == "fam"].pivot_table(index="family", columns="model", values="family_recall_val_control", aggfunc="mean").round(3)
fam.to_csv(OUT + "M4_ciciot_family_recall.csv")
cl = rd(R2 + "full/seed_*/cleaner.csv")
cl["precision"] = cl.flagged_flipped / cl.n_flagged.replace(0, np.nan); cl["recall"] = cl.flagged_flipped / cl.n_flipped.replace(0, np.nan)
cl["discarded_correct"] = cl.n_flagged - cl.flagged_flipped
cl.groupby("condition")[["n_flipped", "n_flagged", "precision", "recall", "discarded_correct"]].mean().round(3).reindex(order).to_csv(OUT + "M4_ciciot_cleaner.csv")

# bound check, all four datasets (as Fig. 3)
s = base[(base.remedy == "none") & base.model.isin(SUP) & base.kind.isin(["sym", "a2b"])]
g = s.groupby(["dataset", "kind", "rate", "model"]).recall_noisyval_control.mean().reset_index()
g["c"] = [contam(k, r, PI[d]) for d, k, r in zip(g.dataset, g.kind, g.rate)]
g["ceiling"] = np.minimum(1, 0.01 / g.c); g["abs_err"] = (g.recall_noisyval_control - g.ceiling).abs()
g.to_csv(OUT + "bound_points_all.csv", index=False)

# ---------------------------------------------------------------- M1: score-dependent noise in V
se = rd(R2 + "select/seed_*/results_select.csv")
sel = se.groupby(["dataset", "rate", "mode"]).agg(c=("c", "mean"), ceiling=("ceiling", "mean"), recall=("recall_noisy_v", "mean"),
                                                   recall_sd=("recall_noisy_v", "std"), above_clean_tau=("share_mislabeled_above_clean_tau", "mean"),
                                                   fpr=("fpr_noisy_v", "mean")).round(3)
sel.to_csv(OUT + "M1_select_summary.csv")
# paired tests: each mode vs random (seed-level mean over the four detectors), Holm within dataset
rows = []
for ds, x in se.groupby("dataset"):
    ps, rec = [], []
    for rate in sorted(x.rate.unique()):
        r0 = x[(x.rate == rate) & (x["mode"] == "random")].groupby("seed").recall_noisy_v.mean()
        for mode in ["low", "soft", "high"]:
            r1 = x[(x.rate == rate) & (x["mode"] == mode)].groupby("seed").recall_noisy_v.mean()
            d = (r1 - r0).dropna(); ps.append(wx(d)); rec.append((ds, rate, mode, r0.mean(), r1.mean(), d.median()))
    for (ds_, rate, mode, a, b, md), p, pa in zip(rec, ps, holm(ps)):
        rows.append({"dataset": ds_, "rate": rate, "mode": mode, "recall_random": a, "recall_mode": b, "median_diff": md, "p": p, "p_holm": pa})
pd.DataFrame(rows).round(4).to_csv(OUT + "M1_select_tests.csv", index=False)

# ---------------------------------------------------------------- M2: cleaners
cc = rd(R2 + "cleaners/seed_*/cleaners.csv")
cc = cc[cc.cleaner != "none"].copy()
cc["precision"] = cc.flagged_flipped / cc.n_flagged.replace(0, np.nan); cc["recall"] = cc.flagged_flipped / cc.n_flipped.replace(0, np.nan)
cc["discarded_correct"] = cc.flagged_correct
cc.groupby(["dataset", "kind", "cleaner"])[["n_flipped", "n_flagged", "precision", "recall", "discarded_correct"]].mean().round(3).to_csv(OUT + "M2_cleaner_accuracy.csv")
rc = rd(R2 + "cleaners/seed_*/results_cleaners.csv")
famr = rc[rc.kind == "fam"].groupby(["dataset", "cleaner"]).family_recall_val_control.agg(ms).unstack(1)
famr.to_csv(OUT + "M2_family_recall_by_cleaner.csv")
allr = rc.groupby(["dataset", "kind", "cleaner"]).recall_val_control.mean().unstack(2).round(3)
allr.to_csv(OUT + "M2_recall_by_cleaner.csv")
rows = []
for ds, x in rc.groupby("dataset"):
    ps, rec = [], []
    for kind, metric in (("fam", "family_recall_val_control"), ("sym", "recall_val_control"), ("a2b", "recall_val_control"), ("b2a", "recall_val_control")):
        y = x[x.kind == kind]
        r0 = y[y.cleaner == "none"].groupby("seed")[metric].mean()
        for clr in ["confident", "ens_majority", "ens_consensus", "coteaching", "coteaching_oracle"]:
            r1 = y[y.cleaner == clr].groupby("seed")[metric].mean(); d = (r1 - r0).dropna()
            ps.append(wx(d)); rec.append((ds, kind, clr, r0.mean(), r1.mean(), d.median()))
    for (ds_, kind, clr, a, b, md), p, pa in zip(rec, ps, holm(ps)):
        rows.append({"dataset": ds_, "kind": kind, "cleaner": clr, "recall_none": a, "recall_cleaner": b, "median_diff": md, "p": p, "p_holm": pa})
pd.DataFrame(rows).round(4).to_csv(OUT + "M2_cleaner_tests.csv", index=False)

# ---------------------------------------------------------------- M5: tuned vs default
tu = rd(R2 + "tuned/seed_*/results_tuned.csv")
k = ["dataset", "seed", "condition", "model"]
m = tu.merge(base[base.remedy == "none"][k + ["recall_val_control", "recall_noisyval_control", "auc_control"]], on=k, suffixes=("_tuned", "_default"))
assert len(m) == len(tu)
summ = m.groupby(["dataset", "kind"])[[c for c in m.columns if c.endswith(("_tuned", "_default"))]].mean().round(3)
summ.to_csv(OUT + "M5_tuned_vs_default.csv")
rows = []
for ds, x in m.groupby("dataset"):
    ps, rec = [], []
    for kind in ["clean", "sym", "a2b", "b2a", "fam"]:
        for metric in ["recall_val_control", "recall_noisyval_control"]:
            y = x[x.kind == kind]; a = y.groupby("seed")[metric + "_default"].mean(); b = y.groupby("seed")[metric + "_tuned"].mean()
            d = b - a; ps.append(wx(d)); rec.append((ds, kind, metric, a.mean(), b.mean(), d.median()))
    for (ds_, kind, metric, a, b, md), p, pa in zip(rec, ps, holm(ps)):
        rows.append({"dataset": ds_, "kind": kind, "metric": metric, "default": a, "tuned": b, "median_diff": md, "p": p, "p_holm": pa})
pd.DataFrame(rows).round(4).to_csv(OUT + "M5_tuned_tests.csv", index=False)
print("done")
