# src/02_granger_group_networks.py
import os, glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

N_ROI = 116
THR_FREQ = 0.70  # paper
FDR_BIN_DIR = "outputs/granger_subject"
OUT_DIR = "outputs/granger_group"

def load_labels(labels_csv="data/labels.csv"):
    lab = pd.read_csv(labels_csv)
    lab["gp"] = lab["gp"].str.lower()
    return dict(zip(lab["subject_id"].astype(str), lab["gp"]))

def load_subject_bin(sid: str):
    fp = f"{FDR_BIN_DIR}/gc_bin_fdr_subject_{sid}.csv"
    M = pd.read_csv(fp, header=None).values.astype(int)
    return M[:N_ROI, :N_ROI]

def edge_frequency(mats):
    stack = np.stack(mats, axis=0)  # S×N×N
    return stack.mean(axis=0)       # frequency (0..1)

def threshold_by_freq(F, thr=THR_FREQ):
    A = (F >= thr).astype(int)
    np.fill_diagonal(A, 0)
    return A

def plot_network(freq, adj_thr, title, out_png):
    G = nx.DiGraph()
    for i in range(N_ROI):
        G.add_node(i)

    # add edges where thr adj=1, weight=freq
    rows, cols = np.where(adj_thr == 1)
    for u, v in zip(rows, cols):
        G.add_edge(u, v, weight=float(freq[u, v]))

    pos = nx.kamada_kawai_layout(G)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=250)
    nx.draw_networkx_edges(G, pos, arrows=True, width=[2 + 6*w for w in weights], alpha=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main(labels_csv="data/labels.csv"):
    os.makedirs(OUT_DIR, exist_ok=True)
    label_map = load_labels(labels_csv)

    # discover subjects from outputs
    files = glob.glob(f"{FDR_BIN_DIR}/gc_bin_fdr_subject_*.csv")
    sids = [os.path.basename(f).replace("gc_bin_fdr_subject_","").replace(".csv","") for f in files]

    nh_mats, hl_mats = [], []
    for sid in sids:
        if sid not in label_map:
            continue
        M = load_subject_bin(sid)
        if label_map[sid] == "nh":
            nh_mats.append(M)
        elif label_map[sid] == "hl":
            hl_mats.append(M)

    F_nh = edge_frequency(nh_mats)
    F_hl = edge_frequency(hl_mats)

    A_nh = threshold_by_freq(F_nh, THR_FREQ)
    A_hl = threshold_by_freq(F_hl, THR_FREQ)

    pd.DataFrame(F_nh).to_csv(f"{OUT_DIR}/NH_edge_frequency.csv", index=False, header=False)
    pd.DataFrame(F_hl).to_csv(f"{OUT_DIR}/HL_edge_frequency.csv", index=False, header=False)
    pd.DataFrame(A_nh).to_csv(f"{OUT_DIR}/NH_thr70_adj.csv", index=False, header=False)
    pd.DataFrame(A_hl).to_csv(f"{OUT_DIR}/HL_thr70_adj.csv", index=False, header=False)

    plot_network(F_nh, A_nh, "NH group-level GC (freq>=0.70)", f"{OUT_DIR}/NH_network_thr70.png")
    plot_network(F_hl, A_hl, "HL group-level GC (freq>=0.70)", f"{OUT_DIR}/HL_network_thr70.png")

if __name__ == "__main__":
    main()
