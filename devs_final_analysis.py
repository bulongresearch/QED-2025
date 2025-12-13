import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

plt.rcParams.update({

    "text.usetex": False,
    "mathtext.fontset": "stix",        
    
    # Font sizes
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    
    # Figure aesthetics
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    
    # Legend
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": True,
    
    # Lines
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    
    # Savefig
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# methods & better names (using mathtext for formatting :))
METHODS = {
    "A_no_filter_weak": "Naïve self-training",
    "B_high_surprisal_weak": "High-surprisal filter",
    "C_goldilocks_weak": "Goldilocks filter",
    "I_diverse_ensemble_accumulate": r"$\mathbf{DEVS}$ (ensemble + accumulate)",
    "K_diverse_ensemble_no_accumulate": r"$\mathbf{DEVS}$ (ensemble only)",
    "J_ensemble_naive": "Ensemble self-training (no verification)",
}

# colors for each method
COLORS = {
    "A_no_filter_weak": "#E24A33",                 # muted red
    "B_high_surprisal_weak": "#F5A623",            # warm orange
    "C_goldilocks_weak": "#8E6BBE",                # soft purple
    "I_diverse_ensemble_accumulate": "#348ABD",    # nice blue
    "K_diverse_ensemble_no_accumulate": "#50A14F", # forest green
}

# marker styles
MARKERS = {
    "A_no_filter_weak": "o",
    "B_high_surprisal_weak": "s",
    "C_goldilocks_weak": "^",
    "I_diverse_ensemble_accumulate": "D",
    "K_diverse_ensemble_no_accumulate": "v",
}



def load_runs_for_method(method_key: str):
    # loads everything
    dfs = []
    for seed in SEEDS:
        fname = f"results_{method_key}_seed{seed}.csv"
        if not os.path.exists(fname):
            print(f"[WARN] Missing file: {fname}")
            continue
        df = pd.read_csv(fname)
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        print(f"[WARN] No files found for method {method_key}")
        return None

    return pd.concat(dfs, ignore_index=True)


def aggregate_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grp = df.groupby("generation")[metric]
    mean = grp.mean()
    std = grp.std(ddof=1)
    count = grp.count().clip(lower=1)
    
    # standard error
    se = std / np.sqrt(count)
    
    # 95% CI using t-distribution 
    t_crit = stats.t.ppf(0.975, df=count - 1)
    ci_95 = t_crit * se
 
    return pd.DataFrame({
        "generation": mean.index.values,
        "mean": mean.values,
        "ci": ci_95.values,
    })


def setup_figure(title: str, ylabel: str, xlabel: str = "Generation", 
                 ylim=None, figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    #random spine stuff
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    
    return fig, ax


def plot_generic_metric(metric: str, ylabel: str, title: str,
                        ylim=None, outname: str | None = None):
    fig, ax = setup_figure(title, ylabel, ylim=ylim)

    for key, label in METHODS.items():
        df = load_runs_for_method(key)
        if df is None:
            continue

        agg = aggregate_metric(df, metric)
        color = COLORS.get(key)
        marker = MARKERS.get(key, "o")

        ax.errorbar(
            agg["generation"],
            agg["mean"],
            yerr=agg["ci"],
            label=label,
            marker=marker,
            capsize=4,
            capthick=1.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1.5,
            zorder=3,
        )

    ax.legend(loc="best", frameon=True, facecolor="white")
    
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    if outname is not None:
        plt.savefig(outname)
        print(f"Saved: {outname}")
    plt.close()


def plot_surprisal():
    fig, ax = setup_figure(
        title=r"Test Surprisal (median with $p_{10}$–$p_{90}$ band)",
        ylabel="Surprisal on real test set"
    )

    for key, label in METHODS.items():
        df = load_runs_for_method(key)
        if df is None:
            continue

        agg_med = aggregate_metric(df, "median_surprisal_test")
        agg_p10 = aggregate_metric(df, "p10_surprisal_test")
        agg_p90 = aggregate_metric(df, "p90_surprisal_test")

        color = COLORS.get(key)
        marker = MARKERS.get(key, "o")

        # median line with error bars
        ax.errorbar(
            agg_med["generation"],
            agg_med["mean"],
            yerr=agg_med["ci"],
            label=label,
            marker=marker,
            capsize=4,
            capthick=1.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1.5,
            zorder=3,
        )

        # p10–p90 shaded band
        band = agg_p10[["generation", "mean"]].merge(
            agg_p90[["generation", "mean"]],
            on="generation",
            suffixes=("_p10", "_p90"),
        )
        ax.fill_between(
            band["generation"],
            band["mean_p10"],
            band["mean_p90"],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    ax.legend(loc="best", frameon=True, facecolor="white")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig("plot_surprisal_median_band.png")
    print("Saved: plot_surprisal_median_band.png")
    plt.close()

# Summary statistics computation
def compute_final_generation_summary(
    methods=METHODS,
    seeds=SEEDS,
    outname="final_generation_summary.csv",
    metrics=[
        "acc_all",
        "acc_head",
        "acc_tail",
        "synthetic_label_error",
        "diversity_entropy",
        "n_synth_filtered",
        "median_surprisal_test",
        "p10_surprisal_test",
        "p90_surprisal_test",
    ]
):
    rows = []

    for method_key, method_name in methods.items():
        df = load_runs_for_method(method_key)
        if df is None:
            continue

        # Identify final generation number
        final_gen = df["generation"].max()
        df_final = df[df["generation"] == final_gen]

        row = {
            "method_key": method_key,
            "method_name": method_name,
            "generation_final": final_gen,
        }

        for metric in metrics:
            values = df_final[metric].dropna().values
            n = len(values)

            if n == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_se"] = None
                row[f"{metric}_ci95"] = None
                continue

            mean = np.mean(values)
            std = np.std(values, ddof=1)
            se = std / np.sqrt(n)

            if n > 1:
                tcrit = stats.t.ppf(0.975, df=n-1)
                ci = tcrit * se
            else:
                ci = None

            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95"] = ci

            # Optional extra info
            # row[f"{metric}_min"] = values.min()
            # row[f"{metric}_max"] = values.max()

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outname, index=False)
    print(f"Saved final-generation summary to {outname}")

    return summary_df

def compute_start_end_differences(metrics: list[str], outname: str = "summary_differences.csv"):
    rows = []
    
    for method_key, method_name in METHODS.items():
        df = load_runs_for_method(method_key)
        if df is None:
            continue
        
        gen_min = df["generation"].min()
        gen_max = df["generation"].max()
        
        seed_diffs = {metric: [] for metric in metrics}
        
        for seed in df["seed"].unique():
            seed_df = df[df["seed"] == seed]
            
            start_row = seed_df[seed_df["generation"] == gen_min]
            end_row = seed_df[seed_df["generation"] == gen_max]
            
            if start_row.empty or end_row.empty:
                continue
            
            for metric in metrics:
                start_val = start_row[metric].values[0]
                end_val = end_row[metric].values[0]
                
                if pd.notna(start_val) and pd.notna(end_val):
                    seed_diffs[metric].append(end_val - start_val)
        
        row = {
            "method_key": method_key,
            "method_name": method_name,
            "gen_start": gen_min,
            "gen_end": gen_max,
        }
        
        for metric in metrics:
            diffs = seed_diffs[metric]
            if len(diffs) > 0:
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs, ddof=1)
                n = len(diffs)
                se = std_diff / np.sqrt(n)
                t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan
                ci = t_crit * se if n > 1 else np.nan
                
                row[f"{metric}_mean_diff"] = mean_diff
                row[f"{metric}_ci"] = ci
                row[f"{metric}_n_seeds"] = n
            else:
                row[f"{metric}_mean_diff"] = np.nan
                row[f"{metric}_ci"] = np.nan
                row[f"{metric}_n_seeds"] = 0
        
        rows.append(row)
    
    summary_diff_df = pd.DataFrame(rows)
    summary_diff_df.to_csv(outname, index=False)
    print(f"Saved: {outname}")
    
    for _, row in summary_diff_df.iterrows():
        print(f"\n{row['method_name']} (gen {int(row['gen_start'])} → {int(row['gen_end'])}):")
        for metric in metrics:
            mean_diff = row.get(f"{metric}_mean_diff", np.nan)
            ci = row.get(f"{metric}_ci", np.nan)
            if pd.notna(mean_diff):
                sign = "+" if mean_diff > 0 else ""
                ci_str = f" ± {ci:.4f}" if pd.notna(ci) else ""
                print(f"  {metric}: {sign}{mean_diff:.4f}{ci_str}")
    
    return summary_diff_df

if __name__ == "__main__":
    print("Generating publication-quality plots...\n")
    
    plot_generic_metric(
        metric="acc_all",
        ylabel="Test accuracy (overall)",
        title="Test Accuracy (Overall)",
        ylim=(0.3, 0.7),
        outname="plot_acc_all.png",
    )

    plot_generic_metric(
        metric="acc_head",
        ylabel="Test accuracy (head / easy cases)",
        title="Test Accuracy (Head / Easy Cases)",
        ylim=(0.3, 0.7),
        outname="plot_acc_head.png",
    )

    plot_generic_metric(
        metric="acc_tail",
        ylabel="Test accuracy (tail / hard cases)",
        title="Test Accuracy (Tail / Hard Cases)",
        ylim=(0.3, 0.7),
        outname="plot_acc_tail.png",
    )

    plot_generic_metric(
        metric="synthetic_label_error",
        ylabel="Synthetic label error rate",
        title="Synthetic Label Error Rate",
        ylim=(0.0, 0.6),
        outname="plot_synth_label_error.png",
    )

    plot_generic_metric(
        metric="diversity_entropy",
        ylabel="Diversity entropy (class distribution)",
        title="Diversity of Synthetic Data (Entropy)",
        outname="plot_diversity_entropy.png",
    )

    plot_generic_metric(
        metric="n_synth_filtered",
        ylabel="# synthetic examples used",
        title="Synthetic Data Volume per Generation",
        outname="plot_n_synth_filtered.png",
    )

    plot_surprisal()

    compute_final_generation_summary()
    compute_start_end_differences(
        metrics=[
            "acc_all",
            "acc_head", 
            "acc_tail",
            "synthetic_label_error",
            "diversity_entropy",
        ],
        outname="summary_differences.csv",
    )


    print("\n All plots saved successfully!")
