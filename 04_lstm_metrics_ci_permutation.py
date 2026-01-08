# src/04_lstm_metrics_ci_permutation.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

def bootstrap_ci(y, p, metric_fn, n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except:
            pass
    vals = np.array(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

def perm_test(y, p, metric_fn, n_perm=2000, seed=7):
    rng = np.random.default_rng(seed)
    obs = metric_fn(y, p)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        val = metric_fn(yp, p)
        if val >= obs:
            cnt += 1
    return float(obs), float((cnt+1)/(n_perm+1))

def main(pred_csv="outputs/lstm/fold_predictions.csv"):
    df = pd.read_csv(pred_csv)
    y = df["y_true"].values.astype(int)
    p = df["y_prob"].values.astype(float)

    auroc_fn = lambda yt, pr: roc_auc_score(yt, pr)
    auprc_fn = lambda yt, pr: average_precision_score(yt, pr)

    # balanced accuracy needs a threshold; paper uses threshold search per test.
    # Here we reproduce BA using the *per-fold best threshold* from fold_metrics (recommended),
    # but if unavailable, use 0.5.
    # Minimal exact approach: read fold_metrics and apply per-fold best_thr.
    try:
        fm = pd.read_csv("outputs/lstm/fold_metrics.csv")
        thr_map = dict(zip(fm["fold"].astype(int), fm["best_thr"].astype(float)))
        yhat = np.array([(pr >= thr_map[int(f)] ) for f, pr in zip(df["fold"].values, p)], dtype=int)
        balacc = balanced_accuracy_score(y, yhat)
        balacc_fn = None
    except:
        balacc = balanced_accuracy_score(y, (p >= 0.5).astype(int))
        balacc_fn = None

    # bootstrap CIs
    auroc_m, auroc_lo, auroc_hi = bootstrap_ci(y, p, auroc_fn)
    auprc_m, auprc_lo, auprc_hi = bootstrap_ci(y, p, auprc_fn)

    # permutation p-values
    auroc_obs, auroc_p = perm_test(y, p, auroc_fn)
    auprc_obs, auprc_p = perm_test(y, p, auprc_fn)

    print("AUROC mean [95% CI]:", f"{auroc_m:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]", "perm_p=", f"{auroc_p:.4f}")
    print("AUPRC mean [95% CI]:", f"{auprc_m:.3f} [{auprc_lo:.3f}, {auprc_hi:.3f}]", "perm_p=", f"{auprc_p:.4f}")
    print("Balanced Accuracy (fold-threshold):", f"{balacc:.3f}")

if __name__ == "__main__":
    main()
