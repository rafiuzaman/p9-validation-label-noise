#!/usr/bin/env python3
"""P9 v2.0 diagnostic for the 'select' part (reviewer comment M1), 23 Nov 2026.
Repeats run_seed_select of P9_run_v2.0.py exactly (same data, seeds, training noise and selection of mislabeled
V records) and records, for every dataset, seed, rate, mode and detector, why recall departs from alpha/c:
  - rho = (share of all V attacks scoring above the noisy threshold) / (share of mislabeled V attacks above it).
    The bound alpha/c assumes rho = 1 (mislabeled attacks scored like attacks in general). In general
    recall ~ rho * alpha / c (generalized bound, checked here against the measured recall on C).
  - mean percentile of the mislabeled attacks among all V attacks under the detector's own score
    (0.5 = typical; below 0.5 = low-scoring), and Spearman correlation of reference and detector scores on V attacks;
  - attack-family make-up of the mislabeled set (largest family share and its name).
Run: python P9_select_diag.py --data-dir <vmshare/data> --out <dir> [--seeds ...]
"""
import argparse, importlib.util, os, sys, time, json
import numpy as np, pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p9", os.path.join(here, "P9_run_v2.0.py"))
p9 = importlib.util.module_from_spec(spec); sys.modules["p9"] = p9; spec.loader.exec_module(p9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/P9_v2_cache"))
    ap.add_argument("--seeds", type=int, nargs="+", default=p9.SEEDS)
    ap.add_argument("--datasets", nargs="+", default=p9.ALL_DATASETS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from scipy.stats import spearmanr
    datasets = p9.load_datasets_vm(a.data_dir, a.cache_dir, a.datasets)
    out_f = os.path.join(a.out, "select_diag.csv")
    done = set()
    if os.path.exists(out_f):
        done = set(pd.read_csv(out_f)["seed"].unique())
    for seed in a.seeds:
        if seed in done:
            continue
        rows, t0 = [], time.time()
        for ds in datasets:
            sub, idx_v, idx_c = p9.seed_split(ds, seed)
            X, y, fam = ds["Xtr"][sub], ds["ytr"][sub], ds["ftr"][sub]
            Xv, yv, fv = ds["Xho"][idx_v], ds["yho"][idx_v], ds["fho"][idx_v]
            Xc, yc = ds["Xho"][idx_c], ds["yho"][idx_c]
            rs = np.random.RandomState(seed + 7000)
            half = rs.rand(len(y)) < 0.5
            ref = p9.make_lgbm(seed + 7000, 100).fit(X[half], y[half])
            ref_v = ref.predict_proba(Xv)[:, 1]
            att = np.flatnonzero(yv == 1)
            conds = [c[0] for c in p9.conditions(ds["families"])]
            for rate in p9.VAL_RATES:
                ci = conds.index("a2b_%d" % int(rate * 100))
                y_noisy, _ = p9.corrupt(y, fam, "a2b", rate, "", np.random.RandomState(seed * 1000 + ci))
                models = {m: p9.build_and_fit(m, X, y_noisy, seed) for m in p9.SUPERVISED}
                sv = {m: p9.scores(m, mod, Xv) for m, mod in models.items()}
                sc = {m: p9.scores(m, mod, Xc) for m, mod in models.items()}
                for mode in p9.SELECT_MODES:
                    if mode == "random":
                        yv_noisy, _ = p9.corrupt(yv, None, "a2b", rate, "", np.random.RandomState(seed * 1000 + ci + 500))
                        mis = np.flatnonzero((yv == 1) & (yv_noisy == 0))
                    else:
                        mis = p9.select_mislabeled(ref_v, att, rate, mode, np.random.RandomState(seed * 100 + int(rate * 100)))
                        yv_noisy = yv.copy(); yv_noisy[mis] = 0
                    lab0 = yv_noisy == 0
                    c = len(mis) / lab0.sum()
                    fams = pd.Series(fv[mis]).value_counts(normalize=True)
                    for m in p9.SUPERVISED:
                        t = p9.quantile_threshold(sv[m][lab0], p9.ALPHA)
                        _, tpr, _, _ = p9.rates(sc[m], yc, t)
                        a_above = float((sv[m][att] > t).mean())
                        m_above = float((sv[m][mis] > t).mean()) if len(mis) else np.nan
                        rho = a_above / m_above if m_above and m_above > 0 else np.inf
                        pct = pd.Series(sv[m][att]).rank(pct=True).to_numpy()
                        pos = {j: i for i, j in enumerate(att)}
                        rows.append({"dataset": ds["name"], "seed": seed, "rate": rate, "mode": mode, "model": m,
                                     "c": c, "ceiling": min(1.0, p9.ALPHA / c), "recall": tpr,
                                     "v_attacks_above_t": a_above, "mislabeled_above_t": m_above, "rho": rho,
                                     "generalized_ceiling": min(1.0, rho * p9.ALPHA / c) if np.isfinite(rho) else 1.0,
                                     "mislabeled_pct_mean": float(np.mean([pct[pos[i]] for i in mis])) if len(mis) else np.nan,
                                     "spearman_ref_det": float(spearmanr(ref_v[att], sv[m][att]).correlation),
                                     "top_family": fams.index[0] if len(fams) else "", "top_family_share": float(fams.iloc[0]) if len(fams) else np.nan,
                                     "n_families": int(len(fams))})
            print(f"{time.strftime('%F %T')} seed {seed} {ds['name']} done", flush=True)
        pd.DataFrame(rows).to_csv(out_f, mode="a", header=not os.path.exists(out_f), index=False)
        print(f"{time.strftime('%F %T')} seed {seed} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("SELECT DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
