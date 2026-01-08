# src/01_granger_subject_level.py
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests

N_ROI = 116
ADF_ALPHA = 0.05
FDR_ALPHA = 0.05
AIC_MAXLAG = 10
FINAL_K = 5  # paper: K=5 (lowest AIC for most ROI pairs)

def read_ts(fp: str) -> np.ndarray:
    X = pd.read_csv(fp, header=None).values.astype(float)
    if X.shape[0] > X.shape[1] and X.shape[1] <= 500:
        X = X.T
    return X[:N_ROI, :]

def adf_p(x: np.ndarray) -> float:
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return 1.0

def stationarize_adf_diff_adf(X: np.ndarray, alpha=ADF_ALPHA):
    series, rep = [], []
    minT = X.shape[1]
    for r in range(X.shape[0]):
        ts = np.nan_to_num(X[r], nan=np.nanmedian(X[r]))
        p0 = adf_p(ts)
        diff_applied = 0
        if p0 >= alpha:
            ts = np.diff(ts, 1)  # first-order differencing
            diff_applied = 1
        p1 = adf_p(ts)          # repeated ADF
        minT = min(minT, len(ts))
        series.append(ts)
        rep.append([r, p0, diff_applied, p1])
    Xs = np.stack([s[:minT] for s in series], axis=0)
    rep = pd.DataFrame(rep, columns=["roi_index","adf_p_before","diff_applied","adf_p_after"])
    return Xs, rep

def select_lag_aic_pairs(Xs: np.ndarray, maxlag=AIC_MAXLAG, sample_pairs=300, seed=7) -> int:
    # paper: tested lag 1..10 using AIC; K=5 chosen because lowest AIC for most ROI pairs
    rng = np.random.default_rng(seed)
    N = Xs.shape[0]
    best = []
    for _ in range(sample_pairs):
        i, j = rng.integers(0, N, 2)
        if i == j: 
            continue
        try:
            data = np.column_stack([Xs[i], Xs[j]])  # T×2
            sel = VAR(data).select_order(maxlag)
            k = int(sel.aic.idxmin())
            if 1 <= k <= maxlag:
                best.append(k)
        except Exception:
            pass
    if not best:
        return FINAL_K
    vals, cnt = np.unique(best, return_counts=True)
    return int(vals[np.argmax(cnt)])

def granger_p_matrix(Xs: np.ndarray, K=FINAL_K) -> np.ndarray:
    N = Xs.shape[0]
    P = np.ones((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            try:
                # columns [y, x] => test whether x causes y
                res = grangercausalitytests(np.column_stack([Xs[j], Xs[i]]),
                                            maxlag=K, verbose=False)
                P[i, j] = float(res[K][0]["ssr_ftest"][1])  # F-test pvalue at lag K
            except Exception:
                P[i, j] = 1.0
    return P

def bh_fdr(P: np.ndarray, alpha=FDR_ALPHA):
    mask = ~np.eye(P.shape[0], dtype=bool)
    p = np.clip(P[mask], 0.0, 1.0)
    rej, q, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    Q = np.ones_like(P, dtype=float)
    B = np.zeros_like(P, dtype=int)
    Q[mask] = q
    B[mask] = rej.astype(int)
    return Q, B

def main(data_dir="data/roi_timeseries", out_dir="outputs/granger_subject"):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "subject_*.csv")))
    laglog = []

    for fp in tqdm(files, desc="GC subject-level"):
        sid = os.path.basename(fp).replace("subject_", "").replace(".csv", "")
        X = read_ts(fp)
        Xs, rep = stationarize_adf_diff_adf(X, ADF_ALPHA)

        k_aic = select_lag_aic_pairs(Xs, AIC_MAXLAG)
        K = FINAL_K  # enforce paper K=5
        laglog.append([sid, k_aic, K])

        P = granger_p_matrix(Xs, K)
        Q, Bfdr = bh_fdr(P, FDR_ALPHA)

        pd.DataFrame(P).to_csv(f"{out_dir}/gc_p_subject_{sid}.csv", index=False, header=False)
        pd.DataFrame(Q).to_csv(f"{out_dir}/gc_q_subject_{sid}.csv", index=False, header=False)
        pd.DataFrame(Bfdr).to_csv(f"{out_dir}/gc_bin_fdr_subject_{sid}.csv", index=False, header=False)
        rep.to_csv(f"{out_dir}/stationarity_{sid}.csv", index=False)

    pd.DataFrame(laglog, columns=["subject_id","k_aic_mode_est","k_final"]).to_csv(
        f"{out_dir}/lag_log.csv", index=False
    )

if __name__ == "__main__":
    main()
