# import packages (please include all required packages in this code cell)
import numpy as np
import pandas as pd
from group_lasso import LogisticGroupLasso # type: ignore
import argparse
import re
import scipy.stats as sps   
from pathlib import Path
from tqdm import tqdm
from os import listdir
from functools import reduce

# plots
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# feature selection
from sklearn.feature_selection import *

# base estimators model for embedded methods (we can change them as needed)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import KMeans

# pipelines 
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# metrics
from sklearn.metrics import * # accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
# Configuration
RANDOM_STATE = 42
DATA_PATH = 'data/data-2.csv.gz'
RESULTS_FOLDER = './results'
DEBUG = False # If true, use subset of columns X

# Create figures folder
FIGURES_FOLDER = Path('figures')
FIGURES_FOLDER.mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

def save_figure(fig, name, dpi=300, bbox_inches='tight', pdf=True):
    if pdf:
        fig.savefig(FIGURES_FOLDER / f'{name}.pdf', bbox_inches=bbox_inches)
    fig.savefig(FIGURES_FOLDER / f'{name}.png', dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {name}")


# ==============================================================================
# Load data and basic schema
# ==============================================================================
def load_data():

    global d
    global X
    global y
    global n_trials
    global feature_cols
    global unique_bins
    global inner_cv 
    global outer_cv
    global feature_counts
    global bins_sorted

    d = pd.read_csv(DATA_PATH)

    if DEBUG:
         d = d.iloc[:, :100]

    # Normalize column names like unit_0001_bin_3 -> unit_1_bin_3
    d.columns = d.columns.str.replace(r'unit_0+(\d+)_bin_(\d+)', r'unit_\1_bin_\2', regex=True)
    y = d['label'].values               
    X = d.drop(columns=['label']).values
    print('X:', X.shape, 'y:', y.shape)

    # Shema
    n_trials = d.shape[0]
    feature_cols = [c for c in d.columns if c != 'label']
    bins_sorted = sorted({int(c.split("_")[-1]) for c in feature_cols if "_bin_" in c})
    p_features = len(feature_cols)

    # Parse neuron IDs and bins
    unit_pat = re.compile(r'^unit_(\d+)_bin_(\d+)$')
    units, bins = [], []
    for c in feature_cols:
        m = unit_pat.match(c)
        if m:
            units.append(int(m.group(1)))
            bins.append(int(m.group(2)))
        else:
            units.append(None); bins.append(None)

    unique_units = sorted({u for u in units if u is not None})
    unique_bins = sorted({b for b in bins if b is not None})

    # Strategy for cross validation 
    outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = RANDOM_STATE)

    print(f"Trials: {n_trials} | Features: {p_features} | Neurons: {len(unique_units)} | Time bins: {len(unique_bins)}")
    print(d.head())

    feature_counts = list(np.arange(10, p_features, 200))

# Create figures folder
FIGURES_FOLDER = Path('figures')
FIGURES_FOLDER.mkdir(exist_ok=True)

def save_figure(fig, name, dpi=300, bbox_inches='tight', pdf=True):
    if pdf:
        fig.savefig(FIGURES_FOLDER / f'{name}.pdf', bbox_inches=bbox_inches)
    fig.savefig(FIGURES_FOLDER / f'{name}.png', dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {name}")


# ==============================================================================
# EDA
# ==============================================================================
def dprime(a, b):
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(0.5*(va + vb)) if (va + vb) > 0 else 1e-9
    return (mb - ma) / s

def class_balance():
    counts = d['label'].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(['Left (0)','Right (1)'], [counts.get(0,0), counts.get(1,0)])
    ax.set_title('Class counts')
    ax.set_ylabel('Trials')

    for i,v in enumerate([counts.get(0,0), counts.get(1,0)]):
        ax.text(i, v, f"{v}", ha='center', va='bottom')
    
    save_figure(fig, 'class_balance')
    plt.show()
    plt.close(fig)

    pct_right = 100.0*counts.get(1,0)/counts.sum()
    print(f"Right-class percentage: {pct_right:.1f}%")

def sparsity():
    zero_rate = (X == 0).mean(axis=0)

    fig, ax = plt.subplots()
    ax.hist(zero_rate, bins=50)
    ax.set_xlabel('Per-feature zero fraction')
    ax.set_ylabel('Count of features')
    ax.set_title('Zero inflation across features')
    
    # Save the histogram
    save_figure(fig, 'zero_inflation_histogram')
    
    plt.show()
    plt.close(fig)

    ## Neuron × Bin d′ heatmap
    # Parse neuron and bin indices from column names: unit_i_bin_t
    pat = re.compile(r'^unit_(\d+)_bin_(\d+)$')
    neuron_to_bins, bins_set = {}, set()
    for c in feature_cols:
        m = pat.match(c)
        if m:
            u, b = int(m.group(1)), int(m.group(2))
            neuron_to_bins.setdefault(u, {})[b] = c
            bins_set.add(b)

    bins_sorted = sorted(bins_set)
    neurons_sorted = sorted(neuron_to_bins.keys())

    labels = y.astype(int)
    left_ix = labels == 0
    right_ix = labels == 1

    # Build neuron×bin d′ matrix
    D = np.zeros((len(neurons_sorted), len(bins_sorted)), dtype=float)
    for i, u in enumerate(neurons_sorted):
        for j, b in enumerate(bins_sorted):
            col = neuron_to_bins[u].get(b)
            if col is None:
                D[i, j] = 0.0
            else:
                x = d[col].values.astype(float)
                D[i, j] = dprime(x[left_ix], x[right_ix])

    # Sort neurons by max |d′|
    order = np.argsort(np.max(np.abs(D), axis=1))[::-1]
    D_sorted = D[order]

    # Robust, symmetric color scaling around 0 (ignoring extreme outliers)
    p = float(np.percentile(np.abs(D_sorted), 98))
    vmax = max(p, 1e-6)
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(D_sorted, aspect='auto', interpolation='nearest',
                cmap='coolwarm', norm=norm)
    ax.set_title("Neuron × Bin d′ (sorted by max |d′|)")
    ax.set_xlabel("Bin"); ax.set_ylabel("Neuron (sorted)")
    ax.set_xticks(range(len(bins_sorted)))
    ax.set_xticklabels(bins_sorted)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("d′ (Right − Left)")
    
    save_figure(fig, 'neuron_bin_dprime_heatmap')
    
    plt.show()
    plt.close(fig)

def bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    x = np.asarray(x, dtype=float)
    if rng is None: rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha/2, 1 - alpha/2])
    return float(x.mean()), float(lo), float(hi)

def temporal_statistics():

    # Build bin structure from columns
    feature_cols = [c for c in d.columns if c != "label"]
    bins_sorted = sorted(
        {int(c.split("_")[-1]) for c in feature_cols if "_bin_" in c}
    )
    bin_cols = {
        b: [c for c in feature_cols if c.endswith(f"_bin_{b}")]
        for b in bins_sorted
    }

    # Choice labels
    y = d["label"].astype(int).values
    L = y == 0
    R = y == 1

    # Containers for stats
    rows = []
    pop_mean, pop_lo, pop_hi, fano = [], [], [], []
    left_mean, left_lo, left_hi = [], [], []
    right_mean, right_lo, right_hi = [], [], []
    abs_diff, dprime_bins = [], []

    # Per-bin statistics
    for b in bins_sorted:
        Xb = d[bin_cols[b]].values.astype(float)

        # Population activity: sum over features in each trial
        trial_sum = Xb.sum(axis=1)
        m, lo, hi = bootstrap_ci(trial_sum)
        pop_mean.append(m)
        pop_lo.append(lo)
        pop_hi.append(hi)

        # Fano factor
        fano.append(
            float(np.var(trial_sum, ddof=1) / (np.mean(trial_sum) + 1e-9))
        )

        # Class-conditioned mean activity
        trial_mean = Xb.mean(axis=1)
        mL, loL, hiL = bootstrap_ci(trial_mean[L])
        mR, loR, hiR = bootstrap_ci(trial_mean[R])
        left_mean.append(mL)
        left_lo.append(loL)
        left_hi.append(hiL)
        right_mean.append(mR)
        right_lo.append(loR)
        right_hi.append(hiR)

        # Difference & d'
        abs_diff.append(abs(mR - mL))
        dprime_bins.append(dprime(trial_mean[L], trial_mean[R]))

        rows.append(
            {
                "bin": b,
                "pop_mean_sum": m,
                "pop_mean_sum_lo": lo,
                "pop_mean_sum_hi": hi,
                "fano_factor": fano[-1],
                "left_mean": mL,
                "left_lo": loL,
                "left_hi": hiL,
                "right_mean": mR,
                "right_lo": loR,
                "right_hi": hiR,
                "abs_diff": abs_diff[-1],
                "d_prime": dprime_bins[-1],
            }
        )

    # Summary DataFrame 
    temporal_df = pd.DataFrame(rows).set_index("bin")
    print(temporal_df.round(3))

    # Figure: choice-conditioned trajectories + Fano factor
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Population Timing & Choice-Conditioned Trajectories", fontsize=16)

    # Left vs Right trajectories with 95% CIs
    ax1.plot(
        bins_sorted, left_mean,
        marker="o", linewidth=2, color="tab:blue", label="Left"
    )
    ax1.fill_between(
        bins_sorted, left_lo, left_hi,
        alpha=0.25, color="tab:blue"
    )

    ax1.plot(
        bins_sorted, right_mean,
        marker="o", linewidth=2, color="tab:orange", label="Right"
    )
    ax1.fill_between(
        bins_sorted, right_lo, right_hi,
        alpha=0.25, color="tab:orange"
    )

    ax1.set_xlabel("Time Bin")
    ax1.set_ylabel("Mean spike rate (per trial)")
    ax1.set_xticks(bins_sorted)

    # Fano factor on right axis
    ax2 = ax1.twinx()
    ax2.plot(
        bins_sorted, fano,
        marker="s", linewidth=2, linestyle="-.",
        color="tab:red", label="Fano factor"
    )
    ax2.set_ylabel("Fano factor", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    # Save & show
    save_figure(fig, "population_timing_choice_trajectories")
    plt.show()
    plt.close(fig)

    return temporal_df

def seperability():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(balanced_accuracy_score)

    bin_scores = []
    for b in unique_bins:  # e.g., [0,1,2,3,4]
        cols_b = [c for c in feature_cols if c.endswith(f"_bin_{b}")]
        Xb = d[cols_b].values.astype(float)
        y  = d["label"].astype(int).values

        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clf",   LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=5000, class_weight="balanced"))
        ])
        scores = cross_val_score(pipe, Xb, y, cv=cv, scoring=scorer, n_jobs=-1)
        bin_scores.append((b, float(scores.mean()), float(scores.std())))

    bins, means, stds = zip(*bin_scores)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(bins, means, yerr=stds, capsize=4)

    overall = float(np.mean(means))
    ax.axhline(overall, linestyle="--", linewidth=1)
    ax.text(bins[-1] + 0.1, overall, f"Mean {overall:.3f}", va="center")

    for bar, m, s in zip(bars, means, stds):
        x = bar.get_x() + bar.get_width()/2
        y = m + s + 0.01
        ax.text(x, y, f"{m:.3f} ± {s:.3f}", ha="center", va="bottom")

    top = max(m + s for m, s in zip(means, stds)) + 0.04
    ax.set_ylim(0.0, min(1.0, top))

    ax.set_title("Per-bin discriminability (5-fold CV balanced accuracy)")
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(list(bins))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    
    save_figure(fig, 'per_bin_discriminability')
    
    plt.show()
    plt.close(fig)

    print("\nPer-bin CV balanced accuracy:")
    for b, m, s in bin_scores:
        print(f"  Bin {b}: {m:.3f} ± {s:.3f}")


def create_neuron_df():
    pat = re.compile(r"^unit_(\d+)_bin_(\d+)$")
    neuron_bins, bins_set = {}, set()
    for c in d.columns:
        m = pat.match(c)
        if m:
            u, b = int(m.group(1)), int(m.group(2))
            neuron_bins.setdefault(u, {})[b] = c
            bins_set.add(b)

    bins_sorted = sorted(bins_set) if bins_set else list(range(5))
    neuron_ids  = sorted(neuron_bins.keys())
    n_trials    = int(d.shape[0])

    rows = []
    for u in neuron_ids:
        cols_u = [neuron_bins[u][b] for b in bins_sorted if b in neuron_bins[u]]
        X = d[cols_u].to_numpy(dtype=float)                    
        if X.size == 0:
            continue
        mean_per_bin = X.mean(axis=0)                          
        mean_rate_bin = float(mean_per_bin.mean())
        max_rate_bin  = float(mean_per_bin.max())
        total_spikes  = float(X.sum())
        active_trials = int((X.sum(axis=1) > 0).sum())
        frac_active   = active_trials / n_trials
        sparsity      = float((X == 0).sum() / X.size)
        peak_bin      = int(bins_sorted[int(np.argmax(mean_per_bin))])

        rows.append({
            "unit": u,
            "mean_rate": mean_rate_bin,              
            "max_rate":  max_rate_bin,              
            "total_spikes": total_spikes,
            "active_trials": active_trials,
            "frac_active": frac_active,
            "sparsity": sparsity,
            "peak_bin": peak_bin
        })

    neuron_df = pd.DataFrame(rows).set_index("unit").sort_index()
    return neuron_df, neuron_bins

def neuron_selectivity():
    # Build per-neuron summary and mapping neuron -> {bin: column_name}
    neuron_df, neuron_bins = create_neuron_df()

    # Optional: basic firing statistics
    mean_rate_bin_all = float(neuron_df["mean_rate"].mean())
    max_rate_bin_any  = float(neuron_df["mean_rate"].max())
    silent_neurons    = int((neuron_df["mean_rate"] == 0).sum())
    high_active       = int((neuron_df["mean_rate"] >= 0.5).sum())  # ≥5 Hz

    print("\nNeuron firing statistics:")
    print(f"  Mean firing rate across neurons: {mean_rate_bin_all:.3f} spikes/100ms "
          f"({mean_rate_bin_all*10:.2f} Hz)")
    print(f"  Most active neuron rate (mean):  {max_rate_bin_any:.3f} spikes/100ms "
          f"({max_rate_bin_any*10:.2f} Hz)")
    print(f"  Number of silent neurons: {silent_neurons}")
    print(f"  Number of highly active neurons (≥5 Hz): {high_active}")

    # ---- Compute per-neuron max |d'| across bins ----
    y = d["label"].astype(int).to_numpy()
    L = (y == 0)
    R = (y == 1)

    max_abs_d = []
    # ensure we always consider bins 0..4 even if some are missing
    all_bins = sorted({*neuron_df["peak_bin"].unique(), *[0, 1, 2, 3, 4]})

    for u in sorted(neuron_df.index):
        vals = []
        nbins_u = neuron_bins.get(u, {})
        for b in all_bins:
            col = nbins_u.get(b, None)
            if col is None:
                continue
            x = d[col].to_numpy(dtype=float)
            vals.append(abs(dprime(x[L], x[R])))
        max_abs_d.append(max(vals) if vals else 0.0)

    max_abs_d = np.asarray(max_abs_d, dtype=float)

    # ---- Single figure: two histograms side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Per-Neuron Activity & Selectivity", fontsize=16)

    # Left: histogram of mean firing rates
    ax = axes[0]
    vals = neuron_df["mean_rate"].to_numpy()
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.85)
    ax.axvline(vals.mean(), color="red", linestyle="--", label="Mean")
    ax.set_title("Distribution of Neuron Firing Rates")
    ax.set_xlabel("Mean firing rate (spikes/100 ms)")
    ax.set_ylabel("Number of neurons")
    ax.legend()

    # Right: histogram of max |d'| across bins
    ax = axes[1]
    ax.hist(max_abs_d, bins=40, edgecolor="black", alpha=0.85)
    ax.set_title("Neuron selectivity distribution")
    ax.set_xlabel("Max |d'| across bins")
    ax.set_ylabel("Count of neurons")

    plt.tight_layout()
    save_figure(fig, "neuron_rates_and_selectivity")
    plt.show()
    plt.close(fig)

def correlation_and_redundancy():
    
    # Build neuron_df and neuron_bins from the main dataset
    neuron_df, neuron_bins = create_neuron_df()

    y = d["label"].astype(int).to_numpy()
    mask_left = (y == 0)
    mask_right = (y == 1)
    n_trials = d.shape[0]

    neurons_all = sorted(neuron_bins.keys())

    # d′(neuron, bin) table
    rows = []
    for u in neurons_all:
        row = {"unit": u}
        for b in range(5):
            col = neuron_bins[u][b]         
            x = d[col].to_numpy(dtype=float)
            row[f"dprime_bin{b}"] = dprime(x[mask_left], x[mask_right])
        rows.append(row)

    dprime_df = pd.DataFrame(rows).set_index("unit")

    # Add selectivity and fraction active to neuron_df
    neuron_df["max_abs_d"] = dprime_df.abs().max(axis=1)
    neuron_df["frac_active"] = neuron_df["active_trials"] / n_trials


    # HEATMAP 1: cluster-level correlations within a fixed bin

    np.random.seed(0)

    # 5-bin d′ vector per neuron
    D = dprime_df[[f"dprime_bin{b}" for b in range(5)]].to_numpy()

    # K=5 k-means on d′ profile
    kmeans_H1 = KMeans(n_clusters=5, random_state=0, n_init=10)
    labels_H1 = kmeans_H1.fit_predict(D)

    # Order clusters by mean time of peak |d′|
    absD = np.abs(D)
    peak_bins_from_d = absD.argmax(axis=1)
    cluster_order = np.argsort(
        [peak_bins_from_d[labels_H1 == k].mean() for k in range(5)]
    )
    label_map_H1 = {old: new for new, old in enumerate(cluster_order)}
    labels_H1_ord = np.array([label_map_H1[l] for l in labels_H1])

    # Cluster-mean signals in informative bin (bin 3)
    bin_choice = 3
    cluster_signals = []
    cluster_sizes_H1 = []

    for k in range(5):
        neurons_k = [u for u, lab in zip(neurons_all, labels_H1_ord) if lab == k]
        cluster_sizes_H1.append(len(neurons_k))
        cols_k = [neuron_bins[u][bin_choice] for u in neurons_k]
        Xk = d[cols_k].to_numpy(dtype=float)      # trials × neurons_in_cluster
        signal_k = Xk.mean(axis=1)                # mean across neurons
        cluster_signals.append(signal_k)

    G = np.column_stack(cluster_signals)          # trials × 5 cluster signals
    corr_H1 = np.corrcoef(G, rowvar=False)        # 5×5 cluster–cluster corr

    # Plot Heatmap 1
    fig, ax = plt.subplots(figsize=(5, 4.2))
    labels_H1_plot = [f"C{k+1}\n(n={cluster_sizes_H1[k]})" for k in range(5)]
    im = ax.imshow(corr_H1, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Within-bin (bin 3)\ncluster-level correlations")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels_H1_plot, rotation=45, ha="right")
    ax.set_yticklabels(labels_H1_plot)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    plt.tight_layout()

    save_figure(fig, "cluster_corr_bin3")
    plt.show()
    plt.close(fig)


    # HEATMAP 2: temporal correlations within neuron-type clusters

    # Feature space: firing, reliability, timing, selectivity
    F = neuron_df[["mean_rate", "frac_active", "peak_bin", "max_abs_d"]].to_numpy()

    kmeans_H2 = KMeans(n_clusters=5, random_state=0, n_init=10)
    labels_H2 = kmeans_H2.fit_predict(F)

    # Order clusters by mean max|d′| (least → most selective)
    cluster_means_maxd = [
        neuron_df["max_abs_d"].iloc[labels_H2 == k].mean()
        for k in range(5)
    ]
    order_H2 = np.argsort(cluster_means_maxd)
    label_map_H2 = {old: new for new, old in enumerate(order_H2)}
    labels_H2_ord = np.array([label_map_H2[l] for l in labels_H2])

    # For each cluster, compute 5×5 bin–bin correlation matrix
    corrs_H2 = []
    sizes_H2 = []
    for k in range(5):
        neurons_k = [u for u, lab in zip(neurons_all, labels_H2_ord) if lab == k]
        sizes_H2.append(len(neurons_k))

        Xk_bins = []
        for b in range(5):
            cols_b = [neuron_bins[u][b] for u in neurons_k]
            Xb = d[cols_b].to_numpy(dtype=float)
            Xk_bins.append(Xb.mean(axis=1))       # trials × 1 (mean across neurons)

        Xk = np.column_stack(Xk_bins)            # trials × 5
        corr_k = np.corrcoef(Xk, rowvar=False)   # 5×5
        corrs_H2.append(corr_k)

    # Plot Heatmap 2: 5 panels
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    fig.suptitle(
        "Within-neuron temporal correlations across neuron-type clusters",
        fontsize=14, y=1.08
    )

    vmin, vmax = -1, 1
    for k, ax in enumerate(axes):
        im = ax.imshow(corrs_H2[k], vmin=vmin, vmax=vmax, cmap="coolwarm")

        ax.set_title(f"Cluster C{k+1}\n(n={sizes_H2[k]})", fontsize=10, pad=10)
        ax.set_xticks(range(5))
        ax.set_xticklabels(range(5))

        if k == 0:
            ax.set_yticks(range(5))
            ax.set_yticklabels(range(5))
            ax.set_ylabel("Time Bin", fontsize=10)
        else:
            ax.set_yticks([])

        if k == 2:
            ax.set_xlabel("Time Bin", fontsize=10)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pearson correlation")

    # Adjust spacing
    plt.subplots_adjust(left=0.05, right=0.9, top=0.85,
                        bottom=0.15, wspace=0.1)

    save_figure(fig, "temporal_corr_by_cluster")
    plt.show()
    plt.close(fig)

    # Tiny console sanity check
    print("Heatmap 1 cluster sizes:", cluster_sizes_H1)
    print("Heatmap 2 cluster sizes:", sizes_H2)


def neuron_clustering():
    # Build per-neuron dataframe (and bins dict, unused here but kept for consistency)
    neuron_df, neuron_bins = create_neuron_df()

    # Features for clustering: strength, reliability, timing
    F = neuron_df[['mean_rate', 'active_trials', 'peak_bin']].copy()

    scaler = StandardScaler()
    Z = scaler.fit_transform(F)

    # Choose k by silhouette over k = 2..6
    scores = {}
    for k in range(2, 7):
        lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
        scores[k] = silhouette_score(Z, lab)

    best_k = max(scores, key=scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    original_labels = kmeans.fit_predict(Z)

    # Reorder clusters by mean firing rate (highest → lowest)
    cluster_means = (
        pd.Series(original_labels)
        .groupby(original_labels)
        .apply(lambda x: neuron_df.loc[x.index, 'mean_rate'].mean())
        .sort_values(ascending=False)
    )
    label_map = {old: new for new, old in enumerate(cluster_means.index)}
    neuron_df['cluster'] = pd.Series(original_labels).map(label_map)

    print(
        f"Chosen k={best_k} "
        f"(silhouette={scores[best_k]:.3f})  | "
        f"scores={{k: round(v, 3) for k, v in scores.items()}}"
    )

    # Color palette
    palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    # === 4-panel figure ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Neuron Clustering Analysis', fontsize=14)

    # Panel 1: mean_rate vs active_trials scatter
    ax = axes[0, 0]
    for i in range(best_k):
        mask = neuron_df['cluster'] == i
        ax.scatter(
            neuron_df.loc[mask, 'mean_rate'],
            neuron_df.loc[mask, 'active_trials'],
            s=25, alpha=0.6, c=palette[i],
            label=f'C{i} (n={int(mask.sum())})',
            edgecolors='white', linewidth=0.3
        )
    ax.set_xlabel('Mean Firing Rate (spikes/100ms)')
    ax.set_ylabel('Active Trials')
    ax.set_title('Neuron Activity Clusters')
    ax.legend(fontsize=9)

    # Panel 2: peak timing histogram by cluster
    ax = axes[0, 1]
    bins = sorted(neuron_df['peak_bin'].unique())
    width = 0.8 / best_k
    x = np.arange(len(bins))
    for i in range(best_k):
        counts = [
            ((neuron_df['cluster'] == i) & (neuron_df['peak_bin'] == b)).sum()
            for b in bins
        ]
        ax.bar(x + i * width, counts, width,
               color=palette[i], alpha=0.8, label=f'C{i}')
    ax.set_xticks(x + width * (best_k - 1) / 2)
    ax.set_xticklabels([f'Bin {b}' for b in bins])
    ax.set_xlabel('Time Bin')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Peak Activity Timing')
    ax.legend(fontsize=9)

    # Panel 3: statistics table
    ax = axes[1, 0]
    ax.axis('off')
    stats = neuron_df.groupby('cluster').agg(
        n=('mean_rate', 'count'),
        FR_Mean=('mean_rate', 'mean'),
        Trials_Mean=('active_trials', 'mean'),
        Peak_Bin=('peak_bin',
                  lambda x: int(x.mode()[0]) if len(x.mode()) > 0
                  else int(x.mean()))
    ).round(2)

    rows = []
    for i in range(best_k):
        rows.append([
            f'C{i}',
            f'{int(stats.loc[i, "n"])}',
            f'{stats.loc[i, "FR_Mean"]:.2f}',
            f'{int(stats.loc[i, "Trials_Mean"])}',
            f'{int(stats.loc[i, "Peak_Bin"])}'
        ])

    table = ax.table(
        cellText=rows,
        colLabels=['Cluster', 'N', 'FR (sp/100ms)', 'Trials', 'Peak Bin'],
        cellLoc='center', loc='center',
        colWidths=[0.15, 0.15, 0.25, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('Cluster Statistics', pad=20)

    # Color-code the cluster column
    for i in range(best_k):
        table[(i + 1, 0)].set_facecolor(palette[i])
        table[(i + 1, 0)].set_alpha(0.3)

    # Panel 4: pie chart of cluster sizes
    ax = axes[1, 1]
    sizes = [int((neuron_df['cluster'] == i).sum()) for i in range(best_k)]
    labels = [f'C{i}: {sizes[i] / sum(sizes) * 100:.1f}%' for i in range(best_k)]
    ax.pie(sizes, labels=labels, colors=palette[:best_k], startangle=90)
    ax.set_title('Cluster Distribution')

    plt.tight_layout()
    save_figure(fig, 'neuron_clustering_analysis')
    plt.show()
    plt.close(fig)

    # Text summary
    print(f"\nSummary (k={best_k}, silhouette={scores[best_k]:.3f}):")
    informative_clusters = stats.nlargest(2, 'FR_Mean').index
    informative_pct = (
        stats.loc[informative_clusters, 'n'].sum()
        / stats['n'].sum() * 100
    )
    print(
        f"Informative neurons (C{informative_clusters[0]}, "
        f"C{informative_clusters[1]}): {informative_pct:.1f}%"
    )
    for i in range(best_k):
        print(
            f"  C{i}: n={int(stats.loc[i, 'n'])}, "
            f"{stats.loc[i, 'FR_Mean']:.2f} sp/100ms, "
            f"{int(stats.loc[i, 'Trials_Mean'])} trials, "
            f"peak bin {int(stats.loc[i, 'Peak_Bin'])}"
        )
        
def target_data_qual_checks():

    nan_feat = np.isnan(X).any(axis=0).sum()
    var_feat = X.var(axis=0)

    const_feat = (var_feat == 0).sum()
    near_zero = (var_feat < 1e-8).sum()

    print(f"NaN features: {nan_feat} | Constant features: {const_feat} | Near-zero variance: {near_zero}")

    j = int(np.argsort(var_feat)[-1])
    fig, ax = plt.subplots()

    if hasattr(X, "iloc"):        
        feature_data = X.iloc[:, j]
    else:                         
        feature_data = X[:, j]

    sps.probplot(feature_data, dist="norm", plot=ax)
    ax.set_title("QQ plot (one high-variance feature)")
    
    # Save QQ plot
    save_figure(fig, 'qq_plot_high_variance')
    
    plt.show()
    plt.close(fig)

    
# ==============================================================================
# Functions, Transformers, etc shared across Task 2
# ==============================================================================
class TopKSelector(BaseEstimator, TransformerMixin):

    def __init__(self, importances, k=10000):
        self.importances = importances
        self.k = k
        
    def fit(self, X, y=None):

        self.top_k_idx_ = np.argsort(self.importances)[-self.k:]
        return self
    
    def transform(self, X):
        return X[:, self.top_k_idx_]
    
    def get_feature_names_out(self, input_features=None):
        return np.array([f"feature_{i}" for i in self.top_k_idx_])
    
    
def evaluation_metrics(y_true, y_pred, y_prob):
    '''
    Evaluate the metrics we wish to track across methods
    for Task 2:
    
    1. accuracy
    2. balanaced_accuracy
    3. auc
    4. f1 score
    '''

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob[:, 1]),
        'f1': f1_score(y_true, y_pred)
    }
    
    return metrics

# ==============================================================================
# Task 2.2 Method 1 (feature selecion using random forest feature importances)
# ==============================================================================
def main_random_forest(X_train, y_train, X_test, y_test, best_model = False, save = True):
    '''
    Feature selection using the importance scores created by
    Random Forest. There are 2 branches to this function

    Branch 1 (best_model = False):
        The aim of this branch is to find an unbiased estimate for various metrics
        fixing the number of features a priori. ("If I build a model using the top 10 features by RF importance,
        the expected balanced accuracy is 0.75"). These metrics are evaluated over f folds. The split is done
        across the full dataset (training and testing).

    Branch 2 (best_model = True):
        The aim of this branch is to find an unbiased estimate of metrics for the best model.
        We hold out a seperate test set, and tune the number of features to select on a training set
        using cross validation. 

    Parameters:
    ----------
    X_train : np.ndarray, shape (n_samples_train, n_features)
        Training feature matrix

    y_train : np.ndarray, shape (n_samples_train,)
        Training target vector

    X_test : np.ndarray, shape (n_samples_test, n_features)
        Test feature matrix

    y_test : np.ndarray, shape (n_samples_test,)
        Test target vector

    best_model : bool, default=False
        If False -> branch 1 as described above.
        If True -> branch 2 as described above.

    save : bool, default=True
        Whether to save results to CSV file

    Returns:
    -------
    results: pd.DataFrame
        If best_model = False: dataframe with [fold, feature_count, <metrics>]
        If best_model = True: dataframe with [best_feature_count, <metrics>]
    '''
    
    # Combining training and test data into full dataset for CV
    # This is because we will later split this training data using cross validation
    # for this branch (see docstring for branch definitions)
    if not best_model:
        X_train = np.vstack([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])

    # Defining the model which we use to compute the feature importances
    feature_importance = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Defining the classifier (selected_features -> y)
    clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
    )

    results_list = []

    # We manually loop through the folds. This is because if we were to use 
    # sklearns grid_search, we would have to recompute feature importances for each k.
    # In summary, this is more efficient. 
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):

        # Basic Loggging 
        print(f"\nFold {fold+1}/{outer_cv.n_splits}: Train size={len(train_idx)}, Val size={len(val_idx)}")

        # Spliting the data into train, val. 
        # If branch 1 -> train_fold and val_fold makes up entire dataset
        # If branch 2 -> train_fold and val_fold makes up a subset (test held out).
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Pre-computing the importance scores
        feature_importance.fit(X_train_fold, y_train_fold)
        importances = feature_importance.feature_importances_

        # Initialising the pipeline (select features then classify)
        random_forest_fs_pipeline = Pipeline([
            ('feature_selection', TopKSelector(importances=importances)),
            ('classifier', clf)
        ])

        # Looping through the feature options
        # branch 2: this is hyperparameter tuning.
        # branch 1: this is fixing k a priori.
        for k in feature_counts:

            # Fitting pipeline for k features
            random_forest_fs_pipeline.set_params(feature_selection__k=k)
            random_forest_fs_pipeline.fit(X_train_fold, y_train_fold)

            # Evaluating the metrics on the validation set
            # branch 1: unbiased estimates for k features
            # branch 2: storing results to choose best k
            metrics = evaluation_metrics(
                y_true = y_val_fold,
                y_pred = random_forest_fs_pipeline.predict(X_val_fold),
                y_prob = random_forest_fs_pipeline.predict_proba(X_val_fold)
            )

            # Storing metrics with the fold index and feature count
            results_list.append({
                'fold': fold + 1,
                'feature_count': k,
                **metrics
            })    

    results_df = pd.DataFrame(results_list)

    # If we want to find the best feature count, choose best balanced accuracy across folds
    if best_model:
        avg_results = results_df.groupby('feature_count')['balanced_accuracy'].mean()
        best_k = avg_results.idxmax()  # k with highest mean balanced accuracy

        # Recomputing the importances on the full training set
        feature_importance.fit(X_train, y_train)
        importances = feature_importance.feature_importances_

        # Refitting piepline on entire training set with best number of features
        random_forest_best = Pipeline([
            ('feature_selection', TopKSelector(importances=importances, k = best_k)),
            ('classifier', clf)
        ]).fit(X_train, y_train)

        # Evaluating on the test set
        metrics = evaluation_metrics(
            y_true = y_test,
            y_pred = random_forest_best.predict(X_test),
            y_prob = random_forest_best.predict_proba(X_test)
        )

        # Overwriting the results df (we don't need cross validation results anymore)
        results_df = pd.DataFrame([{
            'best_feature_count': best_k,
            **metrics
        }])

    # Saving the results as a CSV file
    if save:
        save_name = 'random_forest' + ('_best' if best_model else '') + '.csv'
        save_path = Path(RESULTS_FOLDER, save_name)
        results_df.to_csv(save_path, index = False)

    return results_df
# ==============================================================================
# Task 2.2 Method 2 (Lasso logistic regression - L1 penalty)
# ==============================================================================
def lasso_main(X_train, y_train, X_test, y_test, best_model = False, save = True):
    '''
    Feature selection l1 regularisation. There are two branches to this model

    Branch 1 (best_model = False):
        The aim of this branch is to find an unbiased estimate for various metrics
        fixing the number of features a priori. In practice, we can not set the desired feature count so 
        we implicitly control this with the regularisation strength. These metrics are evaluated over f folds. The split is done
        across the full dataset (training and testing).

    Branch 2 (best_model = True):
        The aim of this branch is to find an unbiased estimate of metrics for the best model.
        We hold out a seperate test set, and tune the number of features to select on a training set
        using cross validation. 

    Parameters:
    ----------
    X_train : np.ndarray, shape (n_samples_train, n_features)
        Training feature matrix

    y_train : np.ndarray, shape (n_samples_train,)
        Training target vector

    X_test : np.ndarray, shape (n_samples_test, n_features)
        Test feature matrix

    y_test : np.ndarray, shape (n_samples_test,)
        Test target vector

    best_model : bool, default=False
        If False -> branch 1 as described above.
        If True -> branch 2 as described above.

    save : bool, default=True
        Whether to save results to CSV file

    Returns:
    -------
    results: pd.DataFrame
        If best_model = False: dataframe with [fold, feature_count, inv_reg_strength, <metrics>]
        If best_model = True: dataframe with [best_inv_reg_strength, best_feature_count, <metrics>]
    '''

    # Combining training and test data into full dataset for CV
    # This is because we will later split this training data using cross validation
    # for this branch (see docstring for branch definitions)
    if not best_model:
        X_train = np.vstack([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])

    # Defining the classification pipeline 
    # Scaling -> Lasso Logisitic Regression
    lasso_logit = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lasso_logit)
    ])

    # grid of inverse of regularization strength options 
    # branch 1: fix a priori
    # branch 2: treat as hyperparameter
    inv_reg_strengths = np.logspace(-2, 5, 100)

    # Find the best alpha on the training set using grid search
    # use balanced accuracy as scoring function to select best
    # regularisation strength
    if best_model:

        grid_search = GridSearchCV(
            estimator = lasso_pipeline,
            param_grid = {'classifier__C': inv_reg_strengths},
            refit = True, # refitting the model after finding best C
            n_jobs = 1,
            cv = outer_cv,
            scoring='balanced_accuracy'
        ).fit(X_train, y_train)

        # Get the best estimator (fitted pipeline)
        best_pipeline = grid_search.best_estimator_

        # Get the classifier from the pipeline
        lasso_classifier = best_pipeline.named_steps['classifier']

        # Count non-zero coefficients
        feature_count = np.sum(lasso_classifier.coef_ != 0)

        # Also get best C
        best_C = grid_search.best_params_['classifier__C']

        # Evaluating the metrics on the held out testing set
        metrics = evaluation_metrics(
            y_true = y_test,
            y_pred = grid_search.predict(X_test),
            y_prob = grid_search.predict_proba(X_test)
        )

        # Getting the feature count 
        results_df = pd.DataFrame([{
            'best_feature_count': feature_count,
            'best_inv_reg_strength': best_C,
            **metrics
        }])

        # Saving the results as a CSV file
        if save:
            save_path = Path(RESULTS_FOLDER, 'lasso_best.csv')
            results_df.to_csv(save_path, index = False)

        return results_df

    
    results_list = []
    # Here, we fix regularisation strength a priori
    # We do not use scikit-learns Grid Search as this does not give us the
    # feature count for each C
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):

        print(f"\nFold {fold+1}/{outer_cv.n_splits}: Train size={len(train_idx)}, Val size={len(val_idx)}")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        

        for C in inv_reg_strengths:
            # Setting the regularisation strength and refitting the model
            lasso_pipeline.set_params(classifier__C = C)
            lasso_pipeline.fit(X_train_fold, y_train_fold)

            # Extracting the indices for coefficients which did not shrink to 0
            selected = np.where(lasso_pipeline['classifier'].coef_ != 0)[0]

            # Evaluate metrics on validation set
            metrics = evaluation_metrics(
                y_true = y_val_fold,
                y_pred = lasso_pipeline.predict(X_val_fold),
                y_prob = lasso_pipeline.predict_proba(X_val_fold)
            )

            results_list.append({
                'fold': fold + 1,
                'feature_count': len(selected),
                'inv_reg_strength': C,
                **metrics
            })

    results_df = pd.DataFrame(results_list)

    # Saving the results as a CSV file
    if save:
        save_path = Path(RESULTS_FOLDER, 'lasso.csv')
        results_df.to_csv(save_path, index = False)

    return results_df

# ==============================================================================
# Task 2.2 Method 3 (forward feature selection + AIC)
# ==============================================================================
class ForwardSelection(BaseEstimator, TransformerMixin):

    def __init__(self, score_func, pipeline, max_features):
        self.score_func = score_func
        self.pipeline = pipeline
        self.max_features = max_features

    def fit(self, X, y):
        full_feature_count = X.shape[1]
        selected_features = []
        best_score = -np.inf # global best aic score
        
        # Forward selection from 1 up to k_features
        for k in range(1, self.max_features + 1):
            print(f'Fitting models with {k} features')
            
            # Get remaining features to try adding
            remaining_features = [i for i in range(full_feature_count) 
                                if i not in selected_features]
            
            best_new_feature = None
            best_new_score = -np.inf # local best score for k + 1 features
            
            # Try adding each remaining feature
            for feat in tqdm(remaining_features):
                candidate_features = selected_features + [feat]
                
                # Fit model with candidate feature set
                X_subset = X[:, candidate_features]
                pipeline_candidate = clone(self.pipeline)
                pipeline_candidate.fit(X_subset, y)
                
                # Compute score
                y_prob = pipeline_candidate.predict_proba(X_subset)
                score = self.score_func(X_subset, y, y_prob)
                
                if score > best_new_score:
                    best_new_score = score
                    best_new_feature = feat
            
            print(f'Best score with {k} features: {best_new_score:.4f}')
            
            # Check if score improved (skip for first feature)
            if k > 1 and best_new_score <= best_score:
                print(f'Score did not improve. Stopping at {k-1} features.')
                break
            
            # Add the best feature
            selected_features.append(best_new_feature)
            best_score = best_new_score
        
        # Store selected features as fitted attribute
        self.selected_features_ = np.array(selected_features)
        self.n_features_in_ = X.shape[1]
        
        # Fit final model on selected features
        X_final = X[:, self.selected_features_]
        self.best_pipeline_ = clone(self.pipeline)
        self.best_pipeline_.fit(X_final, y)
        
        # Store classes for sklearn compatibility
        if hasattr(self.best_pipeline_, 'classes_'):
            self.classes_ = self.best_pipeline_.classes_
        
        return self
    
    def predict(self, X):
        """Predict using the best model"""
        X_selected = X[:, self.selected_features_]
        return self.best_pipeline_.predict(X_selected)
    
    def predict_proba(self, X):
        """Predict probabilities using the best model"""
        X_selected = X[:, self.selected_features_]
        return self.best_pipeline_.predict_proba(X_selected)

    
def forward_selection_main(X_train, y_train, X_test, y_test, save = True):

    logit_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    forward_selection = ForwardSelection(
        score_func= neg_aic,
        pipeline = logit_pipeline,
        max_features = 1000
        )
    
    forward_selection.fit(X_train, y_train)

    # Getting the metrics
    metrics = evaluation_metrics(
        y_true = y_test,
        y_pred = forward_selection.predict(X_test),
        y_prob = forward_selection.predict_proba(X_test)
    )

    # Creating the results dataframe
    num_features = len(forward_selection.selected_features_)

    results_df = pd.DataFrame([{
        'feature_count': num_features,
        **metrics
    }])

    if save:
        save_path = Path(RESULTS_FOLDER, 'best_forward_selection.csv')
        results_df.to_csv(save_path, index = False)

    return results_df

def neg_aic(X, y, p):
    n_params = X.shape[1] + 1
    p = p[:, 1]
    log_lik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    aic = -2 * log_lik + 2 * n_params
    return -aic

# ==============================================================================
# Task 2.3 Method 1 (feature selection using mutual information)
# ==============================================================================
def mutual_information_main(X_train, y_train, X_test, y_test, best_model = False, save = True):
    '''
    Feature selection using mutual information, then fitting logistic regression model.

    Branch 1 (best_model = False):
        Find unbiased estimates for models with k features fixed a priori.
        For each k, the regularization strength C is tuned via nested CV to get
        the best possible model with exactly k features.

    Branch 2 (best_model = True):
        Tune both k and C simultaneously on training set via CV,
        then evaluate on held-out test set for unbiased performance estimate.

    Parameters:
    ----------
    X_train : np.ndarray, shape (n_samples_train, n_features)
        Training feature matrix

    y_train : np.ndarray, shape (n_samples_train,)
        Training target vector

    X_test : np.ndarray, shape (n_samples_test, n_features)
        Test feature matrix

    y_test : np.ndarray, shape (n_samples_test,)
        Test target vector

    best_model : bool, default=False
        If False -> branch 1 as described above.
        If True -> branch 2 as described above.

    save : bool, default=True
        Whether to save results to CSV file

    Returns:
    -------
    results: pd.DataFrame
        If best_model = False: dataframe with [fold, feature_count, inv_reg_strength, <metrics>]
        If best_model = True: dataframe with [best_inv_reg_strength, best_feature_count, <metrics>]
    '''

    inv_reg_strengths = np.logspace(-3, 3, 7) # [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Combining training and test data into full dataset for CV
    # This is because we will later split this training data using cross validation
    # for this branch (see docstring for branch definitions)
    if not best_model:
        X_train = np.vstack([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])

    # branch 1: stores the unbiased estimates for metrics, fixing feature count a priori
    # branch 2: stores the cross-validation results to later use in selecting best feature count
    results_list = []


    # We manually loop through the folds. This is because if we were to use 
    # sklearns grid_search, we would have to recompute feature importances for each k.
    # In summary, this is more efficient. 
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):

        # Basic logging
        print(f"\nFold {fold+1}/{outer_cv.n_splits}: Train size={len(train_idx)}, Val size={len(val_idx)}")

        # Spliting the data into train, val. 
        # If branch 1 -> train_fold and val_fold makes up entire dataset
        # If branch 2 -> train_fold and val_fold makes up a subset (test held out).
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Precomputing the importances for the fold (once per outer fold)
        print("  Computing mutual information scores...")
        importances = mutual_info_classif(
            X = X_train_fold,
            y = y_train_fold,
            discrete_features=True,
            random_state = RANDOM_STATE
        )

        # Defining the pipeline, holds the importance scores for the fold
        mutual_information_logit = Pipeline([
            ('feature_selection', TopKSelector(importances=importances)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])
        
        # Looping through the feature options
        # branch 2: this is hyperparameter tuning.
        # branch 1: this is fixing k a priori.
        for k in feature_counts:     

            # Setting the number of features we want to select
            mutual_information_logit.set_params(feature_selection__k=k)

            # If we want to treat k as a hyperparameter, we don't need an inner
            # fold to tune C, we can just tune (k, C) on the same set.
            if best_model: 
                for C in inv_reg_strengths: 
                    mutual_information_logit.set_params(classifier__C = C)
                    mutual_information_logit.fit(X_train_fold, y_train_fold)

                    # Evaluating on val set
                    metrics = evaluation_metrics(
                        y_true = y_val_fold,
                        y_pred = mutual_information_logit.predict(X_val_fold),
                        y_prob = mutual_information_logit.predict_proba(X_val_fold)
                    )

                    # Append results (cross validation results across grid (K, C))
                    results_list.append({
                        'fold': fold + 1,
                        'feature_count': k,
                        'inv_reg_strength': C,
                        **metrics
                    })

            else: # branch 1
                # We find the best estimator for fixed k
                # Do this by tuning C on inner fold
                grid_search = GridSearchCV(
                    estimator=mutual_information_logit,
                    param_grid={'classifier__C': inv_reg_strengths},
                    cv=inner_cv,
                    scoring='balanced_accuracy',
                    n_jobs=1,
                    refit = True
                ).fit(X_train_fold, y_train_fold)
                
                metrics = evaluation_metrics(
                    y_true = y_val_fold,
                    y_pred = grid_search.predict(X_val_fold),
                    y_prob = grid_search.predict_proba(X_val_fold)
                )
            
                # Append to results list
                results_list.append({
                    'fold': fold + 1,
                    'feature_count': k,
                    'inv_reg_strength': grid_search.best_params_['classifier__C'],
                    **metrics
                })

    results_df = pd.DataFrame(results_list)

    if best_model:
        # Get average balanced accuracy for each (k, C) combination
        average_results = results_df.groupby(['feature_count', 'inv_reg_strength'])['balanced_accuracy'].mean()

        # Get the (k, C) combination with highest average balanced accuracy
        best_k, best_C = average_results.idxmax()  # Returns tuple of (feature_count, C)

        # Recompute importances on full training set
        importances = mutual_info_classif(
            X = X_train,
            y = y_train,
            discrete_features=True,
            random_state = RANDOM_STATE
        )

        # Re-initialize the pipeline with 
        # 1. the importance scores on the full training set
        # 2. the best K found on training set (using cv)
        # 3. the best C found on training set (using cv)
        mutual_information_logit = Pipeline([
            ('feature_selection', TopKSelector(importances=importances, k = best_k)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C = best_C))
        ]).fit(X_train, y_train)

        # Metrics on test set
        metrics = evaluation_metrics(
            y_true = y_test,
            y_pred = mutual_information_logit.predict(X_test),
            y_prob = mutual_information_logit.predict_proba(X_test)
        )

        results_df = pd.DataFrame([{
            'best_feature_count': best_k,
            'best_inv_reg_strength': best_C,
            **metrics
        }])

    # Saving csv
    if save:
        save_name = 'mutual_information' + ('_best' if best_model else '') + '.csv'
        save_path = Path(RESULTS_FOLDER, save_name)
        results_df.to_csv(save_path, index = False)

    return results_df

# ==============================================================================
# Task 2.3 Method 2 (feature selection using group lasso)
# ==============================================================================

def group_lasso_main(X_train, y_train, X_test, y_test, best_model = False, save = True):
    '''
    Feature selection using group l1 regularisation. There are two branches to this model

    Branch 1 (best_model = False):
        The aim of this branch is to find an unbiased estimate for various metrics
        fixing the number of features a priori. In practice, we can not set the desired feature count so 
        we implicitly control this with the regularisation strength. These metrics are evaluated over f folds. The split is done
        across the full dataset (training and testing).

    Branch 2 (best_model = True):
        The aim of this branch is to find an unbiased estimate of metrics for the best model.
        We hold out a seperate test set, and tune the number of features to select on a training set
        using cross validation. 

    Parameters:
    ----------
    X_train : np.ndarray, shape (n_samples_train, n_features)
        Training feature matrix

    y_train : np.ndarray, shape (n_samples_train,)
        Training target vector

    X_test : np.ndarray, shape (n_samples_test, n_features)
        Test feature matrix

    y_test : np.ndarray, shape (n_samples_test,)
        Test target vector

    best_model : bool, default=False
        If False -> branch 1 as described above.
        If True -> branch 2 as described above.

    save : bool, default=True
        Whether to save results to CSV file

    Returns:
    -------
    results: pd.DataFrame
        If best_model = False: dataframe with [fold, feature_count, inv_reg_strength, <metrics>]
        If best_model = True: dataframe with [best_inv_reg_strength, best_feature_count, <metrics>]
    '''

    # Combining training and test data into full dataset for CV
    # This is because we will later split this training data using cross validation
    # for this branch (see docstring for branch definitions)
    if not best_model:
        X_train = np.vstack([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])

    # Creating groups (one for each time bin)
    # [unit_1_bin_1, unit_2_bin_1,...,unit_2238_bin_5] -> [1, 1, ..., 5]
    time_groups = [int(f[-1]) for f in feature_cols]

    reg_options = np.linspace(16, 50, 20)

    gl_classifier = LogisticGroupLasso(
        groups=time_groups, 
        group_reg=0.1,  
        l1_reg=0.0,
        tol=1e-4,
        n_iter = 2_000,
        supress_warning = True,
        scale_reg = 'inverse_group_size'
    )

    gl_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', gl_classifier)
    ])

    if best_model: # brach 2
        # Find the best regularisation strength using the training data
        # done by cross validation
        grid_search = GridSearchCV(
            estimator = gl_pipeline,
            param_grid = {'classifier__group_reg': reg_options},
            refit = True,
            n_jobs = 1,
            cv = outer_cv,
            scoring='balanced_accuracy',
            verbose = 2
        ).fit(X_train, y_train)

        # Deconstucting the best model to get coefficients
        best_pipeline  = grid_search.best_estimator_
        gl_classifier = best_pipeline.named_steps['classifier']
        coefs = gl_classifier.coef_[:, 1]
        best_C = grid_search.best_params_['classifier__group_reg']

        # Getting the norms of the groups coefficients (||beta_1||, ... ||beta_5||)
        groups = np.asarray(time_groups)
        group_ids = np.unique(groups)
        group_norms = {
            f"group_{g}_norm": np.linalg.norm(coefs[groups == g], 2)
            for g in group_ids
        }

        # Metrics on the test set
        metrics = evaluation_metrics(
            y_true = y_test,
            y_pred = best_pipeline.predict(X_test),
            y_prob = best_pipeline.predict_proba(X_test)
        )

        # Creating results dataframe
        results_df = pd.DataFrame([{
            'best_reg_strength': best_C,
            **metrics,
            **group_norms
        }])

        if save:
            save_path = Path(RESULTS_FOLDER, 'best_group_lasso.csv')
            results_df.to_csv(save_path, index = False)

        return results_df

    results_list = []
    # Here, we fix regularisation strength a priori
    # We do not use scikit-learns Grid Search as this does not give us the
    # feature count for each reg strength
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
        
        # basic logging
        print(f"\nFold {fold+1}/{outer_cv.n_splits}: Train size={len(train_idx)}, Val size={len(val_idx)}")

        # Splitting the training data into further training and validation sets
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Fixing the regularisation strength
        for reg in reg_options:
            print(f'  Running group lasso for regularisation = {reg}')
            gl_pipeline.set_params(classifier__group_reg = reg)
            gl_pipeline.fit(X_train_fold, y_train_fold)

            # Metrics on the validation set
            metrics = evaluation_metrics(
                y_true = y_val_fold,
                y_pred = gl_pipeline.predict(X_val_fold),
                y_prob = gl_pipeline.predict_proba(X_val_fold)
            )            

            # Getting the norms of the groups coefficients (||beta_1||, ... ||beta_5||)
            coefs = gl_pipeline['classifier'].coef_[:, 1]
            groups = np.asarray(time_groups)
            group_ids = np.unique(groups)
            group_norms = {
                f"group_{g}_norm": np.linalg.norm(coefs[groups == g], 2)
                for g in group_ids
            }

            # Create result dictionary with norms as columns
            result_dict = {
                'fold': fold + 1,
                'reg_strength': reg,
                **metrics,
                **group_norms
            }

            results_list.append(result_dict)

    results_df = pd.DataFrame(results_list)
            
    if save:
        save_path = Path(RESULTS_FOLDER, 'group_lasso.csv')
        results_df.to_csv(save_path, index = False)

    return results_df

# ==============================================================================
# Task 2.3 Method 3 (feature selection Releif Algorithm)
# ==============================================================================
def relief_score(X, y, n_samples, random_state = 42):
    '''
    Compute the relief score with feature range normalization
    '''

    # Getting the number of rows and columns in X
    n = X.shape[0]
    D = X.shape[1]

    # Setting the random state
    rng = np.random.RandomState(random_state)

    # Setting up sample rows
    sample_indices = rng.choice(n, size = min(n, n_samples), replace=False)

    # Compute feature ranges (max - min for each feature)
    feature_ranges = np.ptp(X, axis=0)  # ptp = "peak to peak" = max - min
    # Handle zero ranges to avoid division by zero
    feature_ranges[feature_ranges == 0] = 1

    # Setting up vector to hold the feature scores
    feature_scores = np.zeros(D)

    for idx in sample_indices:
        x_n = X[idx]
        y_n = y[idx]

        # Get indices excluding the current sample
        same_class_indices = np.where((y == y_n) & (np.arange(n) != idx))[0]
        other_class_indices = np.where((y != y_n) & (np.arange(n) != idx))[0]
    
        # Using Manhattan distance
        same_class_distances = np.sum(np.abs(X[same_class_indices, :] - x_n), axis=1)
        other_class_distances = np.sum(np.abs(X[other_class_indices, :] - x_n), axis=1)

        # Getting the nearest hit and the nearest miss
        nearest_hit  = same_class_indices[np.argmin(same_class_distances)]
        nearest_miss = other_class_indices[np.argmin(other_class_distances)]

        # Update the feature score with range normalization
        feature_scores -= (np.abs(x_n - X[nearest_hit]) / feature_ranges) / len(sample_indices)
        feature_scores += (np.abs(x_n - X[nearest_miss]) / feature_ranges) / len(sample_indices)

    return feature_scores


def relief_main(X_train, y_train, X_test, y_test, best_model = False, save = True):
    '''
    Feature selection using relief algorithm, then fitting logistic regression model.

    Branch 1 (best_model = False):
        Find unbiased estimates for models with k features fixed a priori.
        For each k, the regularization strength C is tuned via nested CV to get
        the best possible model with exactly k features.

    Branch 2 (best_model = True):
        Tune both k and C simultaneously on training set via CV,
        then evaluate on held-out test set for unbiased performance estimate.

    Parameters:
    ----------
    X_train : np.ndarray, shape (n_samples_train, n_features)
        Training feature matrix

    y_train : np.ndarray, shape (n_samples_train,)
        Training target vector

    X_test : np.ndarray, shape (n_samples_test, n_features)
        Test feature matrix

    y_test : np.ndarray, shape (n_samples_test,)
        Test target vector

    best_model : bool, default=False
        If False -> branch 1 as described above.
        If True -> branch 2 as described above.

    save : bool, default=True
        Whether to save results to CSV file

    Returns:
    -------
    results: pd.DataFrame
        If best_model = False: dataframe with [fold, feature_count, inv_reg_strength, <metrics>]
        If best_model = True: dataframe with [best_inv_reg_strength, best_feature_count, <metrics>]
    '''
    inv_reg_strengths = np.logspace(-3, 3, 7) # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # Combining training and test data into full dataset for CV
    # This is because we will later split this training data using cross validation
    # for this branch (see docstring for branch definitions)
    if not best_model:
        X_train = np.vstack([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])

    
    # branch 1: stores the unbiased estimates for metrics, fixing feature count a priori
    # branch 2: stores the cross-validation results to later use in selecting best feature count
    results_list = []
    
    # We manually loop through the folds. This is because if we were to use 
    # sklearns grid_search, we would have to recompute feature importances for each k.
    # In summary, this is more efficient. 
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):

        # Basic logging
        print(f"Fold {fold}: Train size={len(train_idx)}, Val size={len(val_idx)}")

        # Spliting the data into train, val. 
        # If branch 1 -> train_fold and val_fold makes up entire dataset
        # If branch 2 -> train_fold and val_fold makes up a subset (test held out).
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]


        # Pre-computing the relief algorithm scores
        print("  Computing Relief scores...")
        importances = relief_score(
            X = X_train_fold,
            y = y_train_fold,
            n_samples = 600,
            random_state=RANDOM_STATE
        )

        relief_pipeline = Pipeline([
            ('feature_selection', TopKSelector(importances)),
            ('Scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])

        # Looping through the feature options
        # branch 2: this is hyperparameter tuning.
        # branch 1: this is fixing k a priori.
        for k in feature_counts:
            relief_pipeline.set_params(feature_selection__k = k)

            # If we want to treat k as a hyperparameter, we don't need an inner
            # fold to tune C, we can just tune (k, C) on the same set.
            if best_model: # no inner fold needed
                for C in inv_reg_strengths:
                    
                    print(f' Fitting model (K, C) = ({k}, {C})')
                    # Fitting model (k, C)
                    relief_pipeline.set_params(classifier__C = C)
                    relief_pipeline.fit(X_train_fold, y_train_fold)

                    # Evaluating on val set
                    metrics = evaluation_metrics(
                        y_true = y_val_fold,
                        y_pred = relief_pipeline.predict(X_val_fold),
                        y_prob = relief_pipeline.predict_proba(X_val_fold)
                    )

                    # Append results
                    results_list.append({
                        'fold': fold + 1,
                        'feature_count': k,
                        'inv_reg_strength': C,
                        **metrics
                    })
        
            else: # branch 1
                # We find the best estimator for fixed k
                # Do this by tuning C on inner fold
                print(f'   Tuning C for k = {k}')
                grid_search = GridSearchCV(
                    estimator=relief_pipeline,
                    param_grid={'classifier__C': inv_reg_strengths},
                    cv=inner_cv,
                    scoring='balanced_accuracy',
                    n_jobs=1,
                    refit = True,
                ).fit(X_train_fold, y_train_fold)
                
                metrics = evaluation_metrics(
                    y_true = y_val_fold,
                    y_pred = grid_search.predict(X_val_fold),
                    y_prob = grid_search.predict_proba(X_val_fold)
                )
            
                # Append to results list
                results_list.append({
                    'fold': fold + 1,
                    'feature_count': k,
                    'inv_reg_strength': grid_search.best_params_['classifier__C'],
                    **metrics
                })

    results_df = pd.DataFrame(results_list)
            
    if best_model:
        # Get average balanced accuracy for each (k, C) combination
        average_results = results_df.groupby(['feature_count', 'inv_reg_strength'])['balanced_accuracy'].mean()

        # Get the (k, C) combination with highest average balanced accuracy
        best_k, best_C = average_results.idxmax()  # Returns tuple of (feature_count, C)

        # Recompute the importances on the full training set
        importances = relief_score(
            X = X_train, 
            y = y_train,
            n_samples = 600,
            random_state = RANDOM_STATE
        )

        # Re-initialize the pipeline with 
        # 1. the importance scores on the full training set
        # 2. the best K found on training set (using cv)
        # 3. the best C found on training set (using cv)
        relief_pipeline = Pipeline([
            ('feature_selection', TopKSelector(importances, k = best_k)),
            ('Scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C = best_C))
        ]).fit(X_train, y_train)


        metrics = evaluation_metrics(
            y_true = y_test,
            y_pred = relief_pipeline.predict(X_test),
            y_prob = relief_pipeline.predict_proba(X_test)
        )

        results_df = pd.DataFrame([{
            'best_feature_count': best_k,
            'best_inv_reg_strength': best_C,
            **metrics
        }])

    if save:
        save_name = 'relief' + ('_best' if best_model else '') + '.csv'
        save_path = Path(RESULTS_FOLDER, save_name)
        results_df.to_csv(save_path, index = False)

    return results_df

def plot_ba_vs_k():
    '''
    Function which plots the feature count vs balanced accuracy for various methods
    '''
    files = listdir(RESULTS_FOLDER)
    method_names = [str(f).replace('.csv', '') for f in files if 'best' not in str(f)]
    result_paths = [Path(RESULTS_FOLDER, f) for f in files if 'best' not in str(f)]

    def norms_to_feature_count(row):
        # Get norm columns from the row's index, not columns
        norm_columns = [col for col in row.index if 'norm' in col]
        feature_count = sum([2238 * (1 if row[col] > 0 else 0) for col in norm_columns])
        return feature_count


    # Reading in the files
    dfs = []
    columns = ['fold', 'feature_count', 'balanced_accuracy', 'method']
    for method_name, p in zip(method_names, result_paths):
        df = pd.read_csv(p)
        # if group lasso, deduce feature count
        if 'group_lasso' in str(p):
            df['feature_count'] = df.apply(norms_to_feature_count, axis=1)

        df['method'] = method_name
        df = df.loc[:, columns]
        dfs.append(df)

    df = pd.concat(dfs)

    # After concatenating dfs
    # Bin feature counts to group similar values
    df['feature_count_binned'] = pd.cut(df['feature_count'], bins=20)  # or specify custom bins
    df_agg = df.groupby(['feature_count_binned', 'method'], observed=True)['balanced_accuracy'].agg(['mean', 'std']).reset_index()
    df.to_csv('test.csv', index = False)
    # Get bin centers for plotting
    df_agg['feature_count'] = df_agg['feature_count_binned'].apply(lambda x: x.mid)
    df_agg = df_agg.rename(columns = {'mean': 'mean balanced accuracy', 'feature_count': 'feature count'})

    # Set figure size and font
    plt.figure(figsize=(5.5, 2.75))
    plt.rcParams.update({'font.size': 10})

    sns.lineplot(
        data=df_agg,
        x='feature count',
        y='mean balanced accuracy',
        hue='method'
    )

    plt.tight_layout()
    plt.savefig('feature_selection_plot.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE) 

    parser = argparse.ArgumentParser(description='Feature selection on high-dimensional neural spike data')

    parser.add_argument(
        'task',
        type=str,
        choices=['random_forest', 'lasso', 'mutual_information', 'group_lasso', 'relief', 'forward_selection', 'plot_balanced_accuracy'],
        help='Which task to run'
    )

    parser.add_argument(
        '--best_model',
        action='store_true',  # Makes it a flag (present = True, absent = False)
        help='If set, runs branch 2 (tune hyperparameters and evaluate on test set). Otherwise runs branch 1 (exploratory analysis).'
    )

    args = parser.parse_args()

    print(f'Running task: {args.task}')
    print(f'Best model mode: {args.best_model}')

    if args.task == 'lasso':
        print(lasso_main(X_train, y_train, X_test, y_test, best_model=args.best_model))

    elif args.task == 'random_forest':
        print(main_random_forest(X_train, y_train, X_test, y_test, best_model=args.best_model))

    elif args.task == 'relief':
        print(relief_main(X_train, y_train, X_test, y_test, best_model=args.best_model))

    elif args.task == 'mutual_information':
        print(mutual_information_main(X_train, y_train, X_test, y_test, best_model=args.best_model))

    elif args.task == 'group_lasso':
        print(group_lasso_main(X_train, y_train, X_test, y_test, best_model=args.best_model))

    elif args.task == 'forward_selection':
        print(forward_selection_main(X_train, y_train, X_test, y_test, best_model=args.best_model))

    elif args.task == 'plot_balanced_accuracy':
        plot_ba_vs_k()







