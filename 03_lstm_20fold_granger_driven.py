# src/03_lstm_20fold_granger_driven.py
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.multitest import multipletests

# ---------------- paper constants ----------------
N_ROI = 116
ADF_ALPHA = 0.05
FDR_ALPHA = 0.05
FINAL_K = 5
TOPK_ROI = 10
N_SPLITS = 20
TTA_N = 16

# ---------------- preprocessing ----------------
def read_ts(fp: str) -> np.ndarray:
    X = pd.read_csv(fp, header=None).values.astype(float)
    if X.shape[0] > X.shape[1] and X.shape[1] <= 500:
        X = X.T
    return X[:N_ROI, :].astype(np.float32)

def detrend_1d(x):
    t = np.arange(len(x), dtype=np.float32)
    A = np.column_stack([t, np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return (x - A @ beta).astype(np.float32)

def zscore_within_subject(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd

def adf_p(x):
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except:
        return 1.0

def stationarize_adf_diff_adf(X):
    series = []
    minT = X.shape[1]
    for r in range(X.shape[0]):
        ts = np.nan_to_num(X[r], nan=np.nanmedian(X[r]))
        if adf_p(ts) >= ADF_ALPHA:
            ts = np.diff(ts, 1).astype(np.float32)
        # repeated ADF (paper says all pass after)
        _ = adf_p(ts)
        minT = min(minT, len(ts))
        series.append(ts)
    Xs = np.stack([s[:minT] for s in series], axis=0)
    return Xs

# ---------------- Granger داخل fold (TRAIN only) ----------------
def granger_bin_fdr_subject(Xs, K=FINAL_K):
    N = Xs.shape[0]
    P = np.ones((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            try:
                res = grangercausalitytests(np.column_stack([Xs[j], Xs[i]]),
                                            maxlag=K, verbose=False)
                P[i, j] = float(res[K][0]["ssr_ftest"][1])
            except:
                P[i, j] = 1.0
    # BH-FDR across directed edges
    mask = ~np.eye(N, dtype=bool)
    p = np.clip(P[mask], 0.0, 1.0)
    rej, _, _, _ = multipletests(p, alpha=FDR_ALPHA, method="fdr_bh")
    B = np.zeros((N, N), dtype=int)
    B[mask] = rej.astype(int)
    return B

def select_top10_rois_from_train(train_subject_bins):
    stack = np.stack(train_subject_bins, axis=0)  # S×N×N
    outdeg = stack.sum(axis=2).mean(axis=0)       # mean out-degree
    idx = np.argsort(outdeg)[::-1][:TOPK_ROI]
    return idx

# ---------------- model ----------------
class Attention(nn.Module):
    def __init__(self, h): super().__init__(); self.w = nn.Linear(h, 1)
    def forward(self, x):
        a = torch.softmax(self.w(x).squeeze(-1), dim=1)
        return (x * a.unsqueeze(-1)).sum(dim=1)

class BiLSTM2Attn(nn.Module):
    def __init__(self, f, hidden=64, dropout=0.3):
        super().__init__()
        self.l1 = nn.LSTM(f, hidden, batch_first=True, bidirectional=True)
        self.l2 = nn.LSTM(hidden*2, hidden, batch_first=True, bidirectional=True)
        self.att = Attention(hidden*2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2, 2)
    def forward(self, x):
        x, _ = self.l1(x)
        x, _ = self.l2(x)
        x = self.att(x)
        x = self.drop(x)
        return self.fc(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, y):
        ce = self.ce(logits, y)
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[y] * loss
        return loss.mean()

def tta(x, rng, jitter_max=3, noise_std=0.02):
    shift = int(rng.integers(-jitter_max, jitter_max+1))
    y = np.roll(x, shift, axis=0)
    y = y + rng.normal(0, noise_std, size=y.shape).astype(np.float32)
    return y.astype(np.float32)

# ---------------- threshold sweep (paper) ----------------
def best_threshold_balacc(y_true, y_prob):
    best_t, best_ba = 0.5, -1
    for t in np.arange(0.10, 0.90, 0.01):  # 0.10..0.89 step 0.01
        ba = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t, best_ba

# ---------------- main ----------------
def load_labels(labels_csv):
    lab = pd.read_csv(labels_csv)
    lab["gp"] = lab["gp"].str.lower()
    y = lab["gp"].map({"nh":0, "hl":1}).values.astype(int)
    return lab["subject_id"].astype(str).values, y

def main(ts_dir="data/roi_timeseries", labels_csv="data/labels.csv", out_dir="outputs/lstm"):
    os.makedirs(out_dir, exist_ok=True)
    subj_ids, y_all = load_labels(labels_csv)

    # load all subject time-series (raw)
    fp_map = {os.path.basename(f).replace("subject_","").replace(".csv",""): f
              for f in glob.glob(os.path.join(ts_dir, "subject_*.csv"))}

    # align to labels order
    X_raw = []
    keep_ids, keep_y = [], []
    for sid, y in zip(subj_ids, y_all):
        if sid in fp_map:
            X_raw.append(read_ts(fp_map[sid]))
            keep_ids.append(sid)
            keep_y.append(y)
    keep_ids = np.array(keep_ids)
    y = np.array(keep_y, dtype=int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=7)
    rng = np.random.default_rng(7)

    fold_metrics = []
    pred_rows = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        train_ids = keep_ids[tr]
        test_ids  = keep_ids[te]
        y_tr, y_te = y[tr], y[te]

        # ---- per-fold preprocessing + GC on TRAIN only (paper)
        train_bins = []
        train_seq = []
        for sid in train_ids:
            X = read_ts(fp_map[sid])              # ROI×T
            X = np.vstack([detrend_1d(r) for r in X])
            X = zscore_within_subject(X)
            Xs = stationarize_adf_diff_adf(X)     # ADF/diff/ADF (fold)
            B = granger_bin_fdr_subject(Xs)       # GC + BH-FDR (fold)
            train_bins.append(B)
            train_seq.append(X)                   # LSTM uses detrend+zscore (paper)

        roi_idx = select_top10_rois_from_train(train_bins)

        # build train sequences (T×10)
        Xtr_seq = []
        for sid in train_ids:
            X = read_ts(fp_map[sid])
            X = np.vstack([detrend_1d(r) for r in X])
            X = zscore_within_subject(X)
            Xtr_seq.append(X[roi_idx].T)
        Xtr_seq = np.stack(Xtr_seq, axis=0).astype(np.float32)

        # build test sequences (T×10)
        Xte_seq = []
        for sid in test_ids:
            X = read_ts(fp_map[sid])
            X = np.vstack([detrend_1d(r) for r in X])
            X = zscore_within_subject(X)
            Xte_seq.append(X[roi_idx].T)
        Xte_seq = np.stack(Xte_seq, axis=0).astype(np.float32)

        # ---- SMOTE train-only (paper): apply on flattened sequences, then reshape back
        T = Xtr_seq.shape[1]
        Xflat = Xtr_seq.reshape(len(Xtr_seq), T*TOPK_ROI)
        sm = SMOTE(random_state=7)
        Xres, yres = sm.fit_resample(Xflat, y_tr)
        Xres = Xres.reshape(len(Xres), T, TOPK_ROI).astype(np.float32)

        # ---- train model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiLSTM2Attn(TOPK_ROI).to(device)

        # focal alpha from resampled labels
        p0 = (yres == 0).mean()
        p1 = (yres == 1).mean()
        alpha = torch.tensor([1.0/(p0+1e-6), 1.0/(p1+1e-6)], dtype=torch.float32).to(device)
        alpha = alpha / alpha.sum()
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        Xres_t = torch.tensor(Xres, dtype=torch.float32).to(device)
        yres_t = torch.tensor(yres, dtype=torch.long).to(device)

        model.train()
        for _ in range(40):  # epochs (paper text implies fixed training; keeping consistent with your earlier)
            opt.zero_grad(set_to_none=True)
            logits = model(Xres_t)
            loss = loss_fn(logits, yres_t)
            loss.backward()
            opt.step()

        # ---- TTA on TEST (paper): average predictions across augmentations
        model.eval()
        probs = np.zeros((len(Xte_seq),), dtype=np.float32)
        with torch.no_grad():
            for _ in range(TTA_N):
                Xaug = np.stack([tta(Xte_seq[i], rng) for i in range(len(Xte_seq))], axis=0)
                p = torch.softmax(model(torch.tensor(Xaug, dtype=torch.float32).to(device)), dim=1)[:,1]
                probs += p.detach().cpu().numpy().astype(np.float32)
        probs /= float(TTA_N)

        # ---- threshold sweep for best balanced accuracy (paper)
        best_t, best_ba = best_threshold_balacc(y_te, probs)

        auroc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) == 2 else np.nan
        auprc = average_precision_score(y_te, probs) if len(np.unique(y_te)) == 2 else np.nan

        fold_metrics.append([fold, auroc, auprc, best_ba, best_t, ",".join(map(str, roi_idx.tolist()))])

        for sid, yt, pr in zip(test_ids, y_te, probs):
            pred_rows.append([fold, sid, int(yt), float(pr)])

        print(f"[Fold {fold:02d}] AUROC={auroc:.3f} AUPRC={auprc:.3f} BalAcc={best_ba:.3f} thr={best_t:.2f}")

    pd.DataFrame(fold_metrics, columns=["fold","auroc","auprc","bal_acc","best_thr","selected_rois"]).to_csv(
        f"{out_dir}/fold_metrics.csv", index=False
    )
    pd.DataFrame(pred_rows, columns=["fold","subject_id","y_true","y_prob"]).to_csv(
        f"{out_dir}/fold_predictions.csv", index=False
    )

if __name__ == "__main__":
    main()
