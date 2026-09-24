from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def parse_eps(path: Path) -> float:
    match = re.search(r"eps([0-9]+(?:\.[0-9]+)?)", path.stem)
    if not match:
        return 100
    return float(match.group(1).split("_")[0])


def load_summary(path: Path, method: str) -> dict[str, float]:
    df = pd.read_csv(path)
    print(f"Loaded {path.name} with {len(df)} records")
    df.columns = df.columns.str.strip()

    active = (df["bid_up"] + df["bid_down"]) > 0.1

    bid_down = df.loc[df["min_margin"].idxmin(), "bid_down"]

    return {
        "eps": parse_eps(path),

        # Profit
        "objective_value": float(df["net_profit_mean"].mean()),
        "objective_value_std": float(df["net_profit_std"].mean()), # Standard error of the mean

        # ✅ Correct feasibility metrics
        "empirical_feasibility": float(df["p90_success_rate"].mean()),#float(feas.mean()),
        "empirical_feasibility_std": float(df["p90_success_rate_std"].mean()),#float(feas.std(ddof=1)),

        # Other metrics
        "total_bid_kw": float((df["bid_up"] + df["bid_down"]).sum() / 1000.0),

        "min_margin": float(df["min_margin"].min()),
        "avg_margin": float(df["min_margin"].mean()),

        "method": method,
    }

def plot_stacked_metrics(summary,colors):
    fig, axes = plt.subplots(
        3, 1,
        figsize=(6, 7),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    metrics = [
         ("total_daily_bid_kw",
         "Total bid volume [MW]")
         ,
        ("objective_value",
         "Objective value [€ / day]"),

        ("empirical_feasibility",
         "Bid success probability [%]"),
    ]

    multipliers = {
        "objective_value": 1.0,
        "empirical_feasibility": 100.0,
        "total_daily_bid_kw": 1.0
    }

    methods = summary.loc[summary["method"] != "LDP", "method"].unique()
    

    for ax, (metric, ylabel) in zip(axes, metrics):

        for method in methods:

            subset = (
                summary[summary["method"] == method]
                .sort_values("eps")
            )

            color = colors[method]

            marker = (
                "o" if method == "CDP60"
                else "s" if method == "CDP90"
                else "D" 
            )

            x = range(len(subset))
            yval = subset[metric].to_numpy() * multipliers[metric]

            label = (
                "CDP P60" if method == "CDP60"
                else "CDP P80" if method == "CDP80"
                else "CDP P90" if method == "CDP90"
                else "NoPE P90" 
            )

            y_std_col = f"{metric}_std"

            ax.plot(
                x,
                yval,
                marker=marker,
                markersize=5,
                linewidth=2,
                color=color[0],
                label=label
            )

            if y_std_col in subset.columns:

                ci90 = 1.645 * subset[y_std_col].to_numpy()
                yerr = ci90 * multipliers[metric]

                ax.fill_between(
                    x,
                    yval - yerr,
                    yval + yerr,
                    color=color[0],
                    alpha=0.12
                )

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")

    # legend only once
    axes[0].legend(
        loc="upper left",
        ncol=2
    )
    axes[1].set_yscale("log")
    axes[0].set_yscale("log")
    axes[2].set_yticks(np.arange(10, 101, 10))
    eps_vals = sorted(summary["eps"].unique())



    axes[-1].set_xticks(range(len(eps_vals)))
    axes[-1].set_xticklabels([
        "No privacy" if eps >= 11
        else f"{eps:.0f}" if eps >= 1
        else f"{eps:.1f}"
        for eps in eps_vals
    ])


    axes[-1].set_xlabel(r"Privacy level $\epsilon$")


    # reverse epsilon axis
    for ax in axes:
        ax.invert_xaxis()

    for ax in [axes[0], axes[2]]:
        ax.grid(which='major', axis='y', linestyle='--', alpha=0.4)
        ax.grid(which='minor', axis='y', linestyle=':', alpha=0.25)
        

    for ax, letter in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(
            0.02, 0.1,
            letter,
            transform=ax.transAxes,
            va="top",
            fontweight="bold"
        )

    fig.savefig(
        OUTPUT_DIR / "stacked_metrics.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        OUTPUT_DIR / "stacked_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def plot_error_calibration(
        eps_vals,
        ldp_flex_error,
        cdp_flex_error_pre,
        cdp_flex_error_post,
        ldp_bias,
        ldp_std,
        cdp_bias_pre,
        cdp_bias_post,
        cdp_std,
        summary
    ):

    fig, axes = plt.subplots(
        4, 1,
        figsize=(6, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    ax1, ax2, ax3, ax4 = axes

    # --------------------------------------------------
    # Colors consistent with existing paper figures
    # --------------------------------------------------

    colors = {
        "CDP": "#6A3D9A",
        "LDP": "#e7298a"
    }

    # ==================================================
    # (a) Flex-state error
    # ==================================================

    print("eps_vals:", eps_vals)

    ax1.plot(
        range(len(eps_vals)),
        cdp_flex_error_pre,
        marker="o",
        linewidth=2,
        markersize=5,
        color=colors["CDP"],
        label="CDP"
    )

    ax1.plot(
        range(len(eps_vals)),
        ldp_flex_error,
        marker="D",
        linewidth=2,
        markersize=5,
        color=colors["LDP"],
        label="LDP"
    )

    ax1.set_ylabel("Flex-state error [%]")

    ax1.text(
        0.02, 0.95,
        "(a)",
        transform=ax1.transAxes,
        va="top",
        fontweight="bold"
    )

    ax1.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.3
    )
    ax1.set_xticks(range(len(eps_vals)))

    # ==================================================
    # (b) Mean positive bias
    # ==================================================

    ax2.plot(
        range(len(eps_vals)),
        cdp_bias_pre,
        marker="o",
        linewidth=2,
        markersize=5,
        color=colors["CDP"],
        label="CDP"
    )

    ax2.plot(
        range(len(eps_vals)),
        ldp_bias,
        marker="D",
        linewidth=2,
        markersize=5,
        color=colors["LDP"],
        label="LDP"
    )

    ax2.set_ylabel("Mean absolute bias")

    ax2.text(
        0.02, 0.95,
        "(b)",
        transform=ax2.transAxes,
        va="top",
        fontweight="bold"
    )

    ax2.grid(
        True,
        linestyle="--",
        alpha=0.3
    )
    ax2.set_xticks(range(len(eps_vals)))

    # ==================================================
    # (c) Average standard deviation
    # ==================================================

    ax3.plot(
        range(len(eps_vals)),
        cdp_std,
        marker="o",
        linewidth=2,
        markersize=5,
        color=colors["CDP"],
        label="CDP"
    )

    ax3.plot(
        range(len(eps_vals)),
        ldp_std,
        marker="D",
        linewidth=2,
        markersize=5,
        color=colors["LDP"],
        label="LDP"
    )
    ax3.set_xticks(range(len(eps_vals)))

    ax3.set_ylabel("Average std")

    # strongly recommended
    ax3.set_yscale("log")

    ax3.text(
        0.02, 0.95,
        "(c)",
        transform=ax3.transAxes,
        va="top",
        fontweight="bold"
    )

    ax3.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.3
    )

    # ==================================================
    # (d) Empirical feasibility
    # ==================================================

    metric = "empirical_feasibility"
    y_label =  "Bid success probability [%]"

    method = "LDP"
    subset = (summary[summary["method"] == method].sort_values("eps", ascending=False))
    y_std_col = f"{metric}_std"

    print([subset[metric].to_numpy()*100][0][1:])

    ax4.plot(
        range(len(subset)-1),
        [subset[metric].to_numpy()*100][0][1:],
        marker="D",
        markersize=5,
        linewidth=2,
        color=colors["LDP"],
        label="Bid reliability",
    )

    if y_std_col in subset.columns:

        ci90 = 1.645 * subset[y_std_col].to_numpy()[1:]
        yerr = ci90 

        ax4.fill_between(
            range(len(subset)-1),
            [subset[metric].to_numpy()*100][0][1:] - yerr,
            [subset[metric].to_numpy()*100][0][1:] + yerr,
            color=colors["LDP"],
            
            alpha=0.12
        )

    method = "CDP90"
    subset = (summary[summary["method"] == method].sort_values("eps", ascending=False))
    y_std_col = f"{metric}_std"

    ax4.plot(
        range(len(subset)-1),
        [subset[metric].to_numpy()*100][0][1:],
        marker="o",
        markersize=5,
        linewidth=2,
        color=colors["CDP"],
        label="CDP - Bid reliability",
    )

    if y_std_col in subset.columns:

        ci90 = 1.645 * subset[y_std_col].to_numpy()[1:]
        yerr = ci90 

        ax4.fill_between(
            range(len(subset)-1),
            [subset[metric].to_numpy()*100][0][1:] - yerr,
            [subset[metric].to_numpy()*100][0][1:] + yerr,
            color=colors["CDP"],
            alpha=0.12
        )

    ax4.set_ylabel(y_label)
    ax4.grid(True, alpha=0.3, linestyle="--")

    ax41 = ax4.twinx()
    method = "LDP"
    subset = (summary[summary["method"] == method].sort_values("eps", ascending=False))
    metric = "total_daily_bid_kw"
    
    ax41.plot(
        range(len(subset)-1),
        [subset[metric].to_numpy()][0][1:],
        marker="D",
        markersize=5,
        linewidth=2,
        color=colors["LDP"],
        linestyle="--",
        label="LDP - Daily bid volume [MW]"
    )

    method = "CDP90"
    subset = (summary[summary["method"] == method].sort_values("eps", ascending=False))
    
    ax41.plot(
        range(len(subset)-1),
        [subset[metric].to_numpy()][0][1:],
        marker="o",
        markersize=5,
        linewidth=2,
        color=colors["CDP"],
        linestyle="--",
        label="CDP - Daily bid volume [MW]"
    )

    # Custom handles for the two quantities
    metric_handles = [
        Line2D([0], [0], color='grey', linestyle='-', linewidth=1.5,
            label='Bid reliability'),
        Line2D([0], [0], color='grey', linestyle='--', linewidth=1.5,
            label='Daily bid volume')
    ]

    ax4.legend(
        ncol=2,
        handles=metric_handles,
        labels=['Bid reliability', 'Daily bid volume'],
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
        loc='lower center',
    )
    ax41.set_ylabel("Total bid volume [MW]")
    ax41.grid(True, alpha=0.3, linestyle="--")
    ax41.set_yticks(range(40, 101, 10))

    # ==================================================
    # Shared x-axis
    # ==================================================

    ax4.set_xticks(range(len(eps_vals)))


    ax4.text(
        0.02, 0.88,
        "(d)",
        transform=ax4.transAxes,
        va="top",
        fontweight="bold"
    )

    # Privacy decreases to the right
    ax4.set_yticks(np.arange(10, 101, 10))
    ax4.set_ylim(10, 102)

    # ==================================================
    # Legend
    # ==================================================

    axes[0].legend(ncol=2,loc="upper left",
        bbox_to_anchor=(0.05, 1),
        frameon=True
    )

    axes[-1].set_xlabel(r"Privacy level $\epsilon$")
    axes[-1].set_xticklabels([
        "No privacy" if eps >= 11
        else f"{eps:.0f}" if eps >= 1
        else f"{eps:.1f}"
        for eps in eps_vals
    ])

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "error_calibration.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        OUTPUT_DIR / "error_calibration.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def make_plots(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # default font settings for plots
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],

        "axes.titlesize": 13,
        "axes.labelsize": 12,

        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "legend.fontsize": 12,

        "figure.titlesize": 14
    })

    summary = summary.rename(columns={
        "total_bid_kw": "total_daily_bid_kw",
        "min_margin": "unbalance_kw",
        "avg_margin": "avg_unbalance_kw"
    })

    colors = {
        "CDP60": ["#7CAE00","#1b9e77"],  #lime + teal-green
        "CDP80": ["#008B8B","#e7298a"],    #blue
        "CDP90": ["#6A3D9A", "#e6ab02"],        # # purple + muted yellow
        "CDP_NoPE": ["#008B8B","#008B8B"],      # brown + neutral gray
    }


    methods = summary["method"].unique()
    plot_stacked_metrics(summary,colors)

    
    
    ldp_flex_error = [1.08,1.66,5.99,9.36,14.01,27.41,61.67,115.73,148.16] #LDP P90
    cdp_flex_error_pre = [-1.89,-1.90,-2.00,-2.10,-2.32,-2.66,-3.19,-4.51,-11.87] # CDP P90
    cdp_flex_error_post = [-1.51,-1.36,-0.83,-0.48,0.08,1.25,5.16,13.35,83.78] # CDP P90

    ldp_bias = [0.046, 0.055, 0.128, 0.210, 0.402, 0.705, 1.483, 2.328, 2.973]
    ldp_std = [0.320, 0.351, 0.540, 0.819, 1.637, 4.29, 15.337, 40.33, 247.900]

    cdp_bias_post = [0.017, 0.019,0.029, 0.034, 0.046, 0.064, 0.119,0.214, 0.950]
    cdp_std = [0.313, 0.332, 0.373, 0.396, 0.430, 0.493, 0.678, 1.033, 3.908]

    cdp_bias_pre = [0.02, 0.024, 0.039, 0.048, 0.066, 0.098, 0.191, 0.366, 1.61]


    #ldp_zscore = [0.052,0.072,0.128,0.136,0.122,0.089,0.054,0.034,0.007] #LDP P90
    #cdp_zscore_pre = [0.012,0.018,0.035,0.042,0.053,0.067,0.096,0.125,0.189] # CDP P90
    #cdp_zscore_post = [0.012,0.019,0.035,0.042,0.053,0.067,0.096,0.125,0.189 ] # CDP P90

    plot_error_calibration(
        eps_vals=sorted(summary["eps"].unique(),reverse=True)[1:],ldp_flex_error=ldp_flex_error,cdp_flex_error_pre=cdp_flex_error_pre,
                    cdp_flex_error_post=cdp_flex_error_post,ldp_bias=ldp_bias,ldp_std=ldp_std,cdp_bias_pre=cdp_bias_pre,cdp_bias_post=cdp_bias_post,cdp_std=cdp_std, summary=summary)
                    #ldp_zscore=ldp_zscore,cdp_zscore_pre=cdp_zscore_pre,cdp_zscore_post=cdp_zscore_post, summary=summary)

def main():
    # --- Standard ε files ---
    CDP60 = list(RESULTS_DIR.glob("SOC_B_bidvalidation_summary_records_eps???_100_CDP_0.6.csv"))
    CDP80 = list(RESULTS_DIR.glob("SOC_B_bidvalidation_summary_records_eps???_100_CDP_0.8_x.csv")) 
    CDP90= list(RESULTS_DIR.glob("SOC_B_bidvalidation_summary_records_eps???_100_CDP_0.9.csv"))
    NoPE90 = list(RESULTS_DIR.glob("SOC_B_bidvalidation_summary_records_eps???_100_CDP_NoPP.csv"))
    LDP = list(RESULTS_DIR.glob("SOC_B_bidvalidation_summary_records_eps???_100_LDP_0.9.csv"))

    # --- Extra files without eps ---
    CDP60_extra = RESULTS_DIR / "SOC_B_bidvalidation_summary_records_100_0.6.csv"
    CDP80_extra = RESULTS_DIR / "SOC_B_bidvalidation_summary_records_100_0.8_x.csv"
    CDP90_extra = RESULTS_DIR / "SOC_B_bidvalidation_summary_records_0.9.csv"
    NoPE90_extra = RESULTS_DIR / "SOC_B_bidvalidation_summary_records_0.9.csv" 
    LDP_extra = RESULTS_DIR / "SOC_B_bidvalidation_summary_records_0.9.csv"

    rows = []

    # --- Load analytical ε files ---
    for path in sorted(CDP60, key=parse_eps):
        rows.append(load_summary(path, method="CDP60"))


    # # # --- Load scenario ε files ---
    for path in sorted(CDP80, key=parse_eps):
        rows.append(load_summary(path, method="CDP80"))

    # --- Load bonferroni ε files ---
    for path in sorted(CDP90, key=parse_eps):
        rows.append(load_summary(path, method="CDP90"))

    #--- Load deterministic ε files ---
    for path in sorted(NoPE90, key=parse_eps):
      rows.append(load_summary(path, method="CDP_NoPE"))

    for path in sorted(LDP, key=parse_eps):
      rows.append(load_summary(path, method="LDP"))

    # --- Load extra analytical file (fake eps = 100) ---
    if CDP60_extra.exists():
        row = load_summary(CDP60_extra, method="CDP60")
        row["eps"] = 12
        rows.append(row)
    else:
        print(f"Warning: Extra analytical file not found at {CDP60_extra}")

    # # # --- Load extra scenario file (fake eps = 100) ---
    if CDP80_extra.exists():
        row = load_summary(CDP80_extra, method="CDP80")
        row["eps"] = 12
        rows.append(row)
    else:
        print(f"Warning: Extra scenario file not found at {CDP80_extra}")

    # --- Load extra bonferroni file (fake eps = 100) ---
    if CDP90_extra.exists():
        row = load_summary(CDP90_extra, method="CDP90")
        row["eps"] = 12
        rows.append(row)
    else:
        print(f"Warning: Extra bonferroni file not found at {CDP90_extra}")

    if NoPE90_extra.exists():
        row = load_summary(NoPE90_extra, method="CDP_NoPE")
        row["eps"] = 12
        rows.append(row)
    else:
        print(f"Warning: Extra deterministic file not found at {NoPE90_extra}")

    if LDP_extra.exists():
        row = load_summary(LDP_extra, method="LDP")
        row["eps"] = 12
        rows.append(row)
    else:
        print(f"Warning: Extra LDP file not found at {LDP_extra}")

    if not rows:
        raise FileNotFoundError("No valid input files found")

    # --- Build summary ---
    summary = pd.DataFrame(rows).sort_values(["method", "eps"]).reset_index(drop=True)

    summary_print = summary[["eps", "method", "objective_value", "objective_value_std", "empirical_feasibility", "empirical_feasibility_std", "total_bid_kw", "min_margin", "avg_margin"]]

    print(summary_print.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    make_plots(summary)

    summary.to_csv(OUTPUT_DIR / "SOC_bidvalidation_comparison.csv", index=False)

    print(f"\nSaved figures and comparison summary to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()