"""
Aggregate SOC bid-validation summaries over
    (a) privacy-noise realizations  ("seed")   and
    (b) future / out-of-sample day realizations ("test").

Expected layout (same file names as before inside each folder):

    results/
        seed1042/
            test0/SOC_B_bidvalidation_summary_records_eps1.0_100_CDP_0.9.csv
            test1/...
            test2/...
        seed1043/
            test0/...
        ...

If a seed folder has no test*/ sub-folders, the seed folder itself is read as a
single test realization (test 0).  Files without eps in the name (no-privacy
reference, Oracle) are looked up in: test folder -> seed folder -> results/.

For every metric x[s, t] (seed s, test realization t) the script reports

    mean       grand mean over all (s, t)
    noise_std  spread caused by the PRIVACY NOISE, test realization held fixed:
               sqrt( mean over t of  Var_s x[s,t] )
    test_std   spread caused by the FUTURE DAYS, noisy training realization
               held fixed:  sqrt( mean over s of  Var_t x[s,t] )
    total_std  plain std over all (s, t) runs

Variances are averaged, then square-rooted (not an average of stds).

Usage:
    python plot/SOC_seed_analysis.py
    python plot/SOC_seed_analysis.py --results-dir results --target 0.9
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NO_PRIVACY_EPS = 12  # same convention as the existing plot script

PREFIX = "SOC_B_bidvalidation_summary_records"

METHOD_PATTERNS = {
    "CDP60":    f"{PREFIX}_eps*_100_CDP_0.6_x.csv",
    "CDP80":    f"{PREFIX}_eps*_100_CDP_0.8_x.csv",
    "CDP90":    f"{PREFIX}_eps*_100_CDP_0.9.csv",
    "CDP90nB":  f"{PREFIX}_eps*_100_CDPnB_0.9.csv",
    "CDP_NoPE": f"{PREFIX}_eps*_100_CDP_NoPP.csv",
    "LDP":      f"{PREFIX}_eps*_100_LDP_0.9.csv",
    "LDPnB":    f"{PREFIX}_eps*_100_LDPnB_0.9.csv",
}

EXTRA_FILES = {
    "CDP60":    f"{PREFIX}_100_0.6_x.csv",
    "CDP80":    f"{PREFIX}_100_0.8_x.csv",
    "CDP90":    f"{PREFIX}_0.9.csv",
    "CDP_NoPE": f"{PREFIX}_0.9.csv",
    "LDP":      f"{PREFIX}_0.9.csv",
    "Oracle":   "SOC_oracle_bidvalidation_summary_records_0.9.csv.csv",
    "NPE":      f"SOC_NPE_bidvalidation_summary_records_0.9.csv",
}

SCALAR_METRICS = [
    "objective_value",
    "empirical_feasibility",     # mean success probability over hours
    "worst_hour_success",
    "hours_below_target",
    "total_bid_kw",
    "min_margin",
    "avg_margin",
    "negative_flow_pct",         # only filled for NEG_FLOW_METHODS
]

# negative-flow percentage is computed only for these methods
NEG_FLOW_METHODS = ["CDP90","NPE", "LDP"]
NEG_FLOW_DENOM = 90 * 24 *2*144        # 90 scenarios x 24 hours x 2 variables x 144 states (up/down)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def parse_eps(path: Path) -> float:
    m = re.search(r"eps([0-9]+(?:\.[0-9]+)?)", path.stem)
    return float(m.group(1)) if m else float(NO_PRIVACY_EPS)


def parse_num(name: str, word: str) -> int:
    m = re.search(rf"{word}[_-]?(\d+)", name)
    return int(m.group(1)) if m else -1

def read_one(path: Path, target: float) -> tuple[dict, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    hourly = df["empirical_feasibility"].to_numpy(dtype=float)  # per hour
    bid_hourly = (df["bid_up"] + df["bid_down"]).to_numpy(dtype=float)  # per hour
    scalars = {
        "objective_value": float(df["net_profit_mean"].mean()),
        "empirical_feasibility": float(df["p90_success_rate"].mean()),
        "worst_hour_success": float(hourly.min()),
        "hours_below_target": int((hourly < target).sum()),
        "total_bid_kw": float((df["bid_up"] + df["bid_down"]).sum()),
        "min_margin": float(df["min_margin"].min()),
        "avg_margin": float(df["min_margin"].mean()),
        "negative_flow_pct": negative_flow_pct(df),
    }
    return scalars, hourly, bid_hourly


def negative_flow_pct(df: pd.DataFrame) -> float:
    """negative_flow / (90*24) * 100, with negative_flow summed over the rows
    (hours) of the file. NaN if the column is missing."""
    if "negative_flow" not in df.columns:
        return float("nan")
    return float(df["negative_flow"].mean() / NEG_FLOW_DENOM * 100.0)


def find_extra(name: str, td: Path, sd: Path, results_dir: Path, test: int) -> Path | None:
    """
    Locate a file that does not depend on the privacy seed (Oracle, no-privacy).
    Search order (first hit wins):
        results/seed<S>/test<T>/        (same folder as the private results)
        results/seed<S>/
        results/test<T>/                (one Oracle file per test realization)
        results/oracle/test<T>/
        results/oracle/                 (one shared file)
        results/                        (one shared file)
        results/NPE/                    (one shared file)
    """
    folders = [td, sd]
    for base in (results_dir, results_dir / "oracle", results_dir / "NPE"):
        if base.is_dir():
            folders += [d for d in sorted(base.glob("test*"))
                        if d.is_dir() and re.fullmatch(r"test[_-]?\d+", d.name)
                        and parse_num(d.name, "test") == test]
    folders += [results_dir / "oracle", results_dir]
    for f in folders:
        if (f / name).exists():
            return f / name
    return None

def collect(results_dir: Path, target: float):
    seed_dirs = sorted(
        [p for p in results_dir.glob("seed*")
         if p.is_dir() and re.fullmatch(r"seed[_-]?\d+", p.name)],
        key=lambda p: parse_num(p.name, "seed"),
    )
    print(f"Found {len(seed_dirs)} seed folders in {results_dir}")
    if not seed_dirs:
        raise FileNotFoundError(f"No seed<N> folders found in {results_dir}")
 
    scalar_rows, hourly_rows = [], []
    shared_extras = set()  # (method) read from a folder shared by several tests
 
    def add(seed, test, method, eps, path):
        scalars, hourly, bid_hourly = read_one(path, target)
        if method not in NEG_FLOW_METHODS:
            scalars["negative_flow_pct"] = float("nan")
        scalar_rows.append(
            {"seed": seed, "test": test, "method": method, "eps": eps, **scalars})
        for h, s in enumerate(hourly):
            hourly_rows.append({"seed": seed, "test": test, "method": method,
                                "eps": eps, "hour": h, "success": s,"bid_kw": bid_hourly[h]})
 
    for sd in seed_dirs:
        seed = parse_num(sd.name, "seed")
        test_dirs = sorted(
            [p for p in sd.glob("test*")
             if p.is_dir() and re.fullmatch(r"test[_-]?\d+", p.name)],
            key=lambda p: parse_num(p.name, "test"),
        )
        if not test_dirs:
            test_dirs = [sd]  # single test realization
 
        for td in test_dirs:
            test = parse_num(td.name, "test") if td != sd else 0
 
            for method, pattern in METHOD_PATTERNS.items():
                for path in sorted(td.glob(pattern), key=parse_eps):
                    add(seed, test, method, parse_eps(path), path)
 
            for method, name in EXTRA_FILES.items():
                path = find_extra(name, td, sd, results_dir, test)
                if path is not None:
                    # shared = one file reused for every test realization
                    if path.parent in (results_dir, results_dir / "oracle"):
                        shared_extras.add(method)
                    add(seed, test, method, float(NO_PRIVACY_EPS), path)
 
    if not scalar_rows:
        raise FileNotFoundError("No matching summary files found")
 
    print(f"Seeds: {[parse_num(d.name, 'seed') for d in seed_dirs]}")
    n_tests = len({r['test'] for r in scalar_rows})
    print(f"Test realizations per seed: {n_tests}")
    if shared_extras and n_tests > 1:
        print("WARNING: no-privacy / Oracle files for "
              f"{sorted(shared_extras)} were read from a shared folder, so their "
              "test_std is 0 by construction. Run them for every test realization.")
    return pd.DataFrame(scalar_rows), pd.DataFrame(hourly_rows)


# --------------------------------------------------------------------------
# variance decomposition
# --------------------------------------------------------------------------
def decompose(df: pd.DataFrame, col: str, keys: list[str]) -> pd.DataFrame:
    """mean, noise_std, test_std, total_std of `col` grouped by `keys`."""
    noise_var = (df.groupby(keys + ["test"])[col].var(ddof=1)
                   .groupby(level=keys).mean())      # across seeds, test fixed
    test_var = (df.groupby(keys + ["seed"])[col].var(ddof=1)
                  .groupby(level=keys).mean())       # across tests, seed fixed
    g = df.groupby(keys)[col]
    return pd.DataFrame({
        "mean": g.mean(),
        "noise_std": np.sqrt(noise_var),
        "test_std": np.sqrt(test_var),
        "total_std": g.std(ddof=1),
        "n_seeds": df.groupby(keys)["seed"].nunique(),
        "n_tests": df.groupby(keys)["test"].nunique(),
    })


def aggregate_scalars(per_run: pd.DataFrame, target: float) -> pd.DataFrame:
    keys = ["method", "eps"]
    parts = []
    for m in SCALAR_METRICS:
        d = decompose(per_run, m, keys)
        parts.append(d[["mean", "noise_std", "test_std", "total_std"]]
                     .add_prefix(f"{m}_"))
    out = pd.concat(parts, axis=1)
    out["n_seeds"] = per_run.groupby(keys)["seed"].nunique()
    out["n_tests"] = per_run.groupby(keys)["test"].nunique()
    out["share_runs_avg_ok"] = (per_run["empirical_feasibility"] >= target) \
        .groupby([per_run[k] for k in keys]).mean()
    out["share_runs_all_hours_ok"] = (per_run["worst_hour_success"] >= target) \
        .groupby([per_run[k] for k in keys]).mean()
    return out.reset_index().sort_values(["method", "eps"], ascending=[True, False])


def aggregate_hourly(hourly: pd.DataFrame, target: float) -> pd.DataFrame:
    keys = ["method", "eps", "hour"]
    d = decompose(hourly, "success", keys)
    bid = decompose(hourly, "bid_kw", keys)[["mean", "noise_std", "test_std"]].add_prefix("bid_")
    d = d.join(bid)          # bid_mean, bid_noise_std, bid_test_std  [kW per hour]
    d["violation_mean"] = 1 - d["mean"]
    d["share_runs_below_target"] = (hourly["success"] < target) \
        .groupby([hourly[k] for k in keys]).mean()
    return d.reset_index().rename(columns={"mean": "success_mean"})


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
def eps_label(e):
    return "no privacy" if e >= NO_PRIVACY_EPS else f"eps={e:g}"


def pm_table(agg, metric, scale=1.0, fmt="{:.1f}"):
    """cell = mean (±noise / ±test)."""
    def cell(r):
        f = lambda v: "n/a" if pd.isna(v) else fmt.format(v * scale)
        return (f"{f(r[f'{metric}_mean'])} "
                f"(±{f(r[f'{metric}_noise_std'])} / ±{f(r[f'{metric}_test_std'])})")
    tmp = agg.copy()
    tmp["cell"] = tmp.apply(cell, axis=1)
    wide = tmp.pivot(index="method", columns="eps", values="cell")
    wide = wide[sorted(wide.columns, reverse=True)]
    wide.columns = [eps_label(e) for e in wide.columns]
    return wide


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
COLORS = {
    # CDP: purple family, dark -> mid -> light
    "CDP90": "#6A3D9A", "CDP90nB": "#e7298a", "CDP90nCB": "#e7298a", 
    "LDP": "#7CAE00", "LDPnB": "#ae9a00", "LDPnCB": "#005931",

    # Standalone, clearly outside both families
    "NPE":     "#D55E00",   # vermillion
    "Oracle":  "#222222",   # near-black
}

MARKERS = {
     "CDP90": "s", "CDP90nB": "D", "CDP90nCB": "H",
     "LDP": "^", "LDPnB": "o", "LDPnCB": "X", "Oracle": "v",
}
LABELS = {
    "CDP90": "CDP",
    "CDP90nB": "CDP-nB","CDP90nCB": "CDP-nCB", "NPE": "NPE P90",
    "LDP": "LDP", "LDPnB": "LDP-nB","LDPnCB": "LDP-nCB", "Oracle": "Oracle",
}
METHOD_ORDER = ["NPE","CDP90", "CDP90nB",  "LDP", "LDPnB", "CDP60", "CDP80"] #"Oracle",


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    })


def eps_tick(e):
    return "No privacy" if e >= NO_PRIVACY_EPS else (f"{e:.0f}" if e >= 1 else f"{e:.1f}")


def draw_bands(ax, x, y, noise, test, color, log=False, ymax=None):
    """Filled = privacy-noise spread, dashed contour = test-day spread."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    noise, test = np.nan_to_num(noise), np.nan_to_num(test)
    lo_n, hi_n = y - noise, y + noise
    lo_t, hi_t = y - test, y + test
    if log:  # keep the bands drawable on a log axis
        floor = y * 0.01
        lo_n, lo_t = np.where(lo_n <= 0, floor, lo_n), np.where(lo_t <= 0, floor, lo_t)
    if ymax is not None:  # e.g. probabilities cannot exceed 100 %
        hi_n, hi_t = np.minimum(hi_n, ymax), np.minimum(hi_t, ymax)
    ax.fill_between(x, lo_n, hi_n, color=color, alpha=0.22, lw=0)
    ax.plot(x, hi_t, color=color, ls=":", lw=1.0, alpha=0.8)
    ax.plot(x, lo_t, color=color, ls=":", lw=1.0, alpha=0.8)


def band_handles(z):
    return [
        Patch(facecolor="grey", alpha=0.3, label=f"Privacy-noise spread (±{z:g}σ)"),
        Line2D([0], [0], color="grey", ls=":", lw=1.0, label=f"Test-day spread (±{z:g}σ)"),
    ]


def plot_comparison(agg, out_dir, eps_list, z, pct_oracle, log):
    """Methods as lines, eps on x, three stacked metrics."""
    metrics = [
        #("total_bid_kw", "Bid volume", "[MW]", True),
        ("objective_value", "Objective", "[€/day]", True),
        ("empirical_feasibility", "Success", "[%]", False),
    ]
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]
    eps_x = {e: i for i, e in enumerate(sorted(eps_list, reverse=True))}

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True,
                             gridspec_kw={"hspace": 0.07})
    for ax, (metric, name, unit, can_norm) in zip(axes, metrics):
        if metric == "empirical_feasibility":
            scale = 100.0
            ax.set_yticks(range(0, 101, 10))
        else:
            scale = 1.0
        ref = agg[(agg["eps"] >= NO_PRIVACY_EPS)]
        # oracle = ref[ref["method"] == "Oracle"]
        # if pct_oracle and can_norm and not oracle.empty:
        #     scale = 100.0 / float(oracle.iloc[0][f"{metric}_mean"])
        #     unit = "[% of oracle]"

        # if not oracle.empty:
        #     ax.axhline(float(oracle.iloc[0][f"{metric}_mean"]) * scale,
        #                color=COLORS["Oracle"], ls="--", lw=1.5, label="Oracle")

        NPE = ref[ref["method"] == "NPE"]
        if not NPE.empty:
            ax.axhline(float(NPE.iloc[0][f"{metric}_mean"]) * scale,
                       color=COLORS["NPE"], ls="-.", lw=1.5, label="NPE")

        npv = ref[ref["method"] == "CDP90"]
        if not npv.empty:
            ax.axhline(float(npv.iloc[0][f"{metric}_mean"]) * scale,
                       color="#9ad0d0", ls=":", lw=1.5, label="Estimated non-private")
        use_log = log and can_norm and not pct_oracle
        for m in methods:
            sub = agg[(agg["method"] == m) & (agg["eps"].isin(eps_x))].sort_values("eps", ascending=False)
            if sub.empty:
                continue
            x = [eps_x[e] for e in sub["eps"]]
            y = sub[f"{metric}_mean"].to_numpy() * scale
            ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], ms=3, lw=1, label=LABELS[m])
            draw_bands(ax, x, y,
                       z * sub[f"{metric}_noise_std"].to_numpy() * scale,
                       z * sub[f"{metric}_test_std"].to_numpy() * scale,
                       COLORS[m], log=use_log,
                       ymax=100.0 if metric == "empirical_feasibility" else None)

        ax.set_ylabel(f"{name} {unit}")
        ax.grid(True, alpha=0.3, ls="--")
        if use_log:
            ax.set_yscale("log")

    axes[-1].set_xticks(list(eps_x.values()))
    axes[-1].set_xticklabels([eps_tick(e) for e in eps_x])
    axes[-1].set_xlabel(r"Privacy level $\epsilon$")
    for ax in axes:
        ax.invert_xaxis()
    for ax, tag in zip(axes, ["(a)", "(b)"]): #, "(c)"
        ax.text(0.02, 0.1, tag, transform=ax.transAxes, va="top", fontweight="bold", fontsize=14)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h + band_handles(z), l + [x.get_label() for x in band_handles(z)],
               loc="upper center", ncol=3, bbox_to_anchor=(0.512, 1.01), frameon=True,
               fontsize=11)
    fig.align_ylabels(axes)
    fig.subplots_adjust(top=0.90)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out_dir / f"comparison_metrics.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def plot_negative_flow(agg, out_dir, eps_list, z):
    """Negative-flow percentage vs epsilon, one line per method (CDP90, LDP)."""
    metric = "negative_flow_pct"
    methods = [m for m in NEG_FLOW_METHODS if m in set(agg["method"])]
    sub_all = agg[agg["method"].isin(methods)]
    if not methods or sub_all[f"{metric}_mean"].isna().all():
        print("No negative_flow data found; skipping negative-flow plot.")
        return
    eps_x = {e: i for i, e in enumerate(sorted(eps_list))}

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for m in methods:
        sub = agg[(agg["method"] == m) & (agg["eps"].isin(eps_x))] \
            .sort_values("eps", ascending=False)
        if sub.empty:
            continue
        x = [eps_x[e] for e in sub["eps"]]
        y = sub[f"{metric}_mean"].to_numpy()
        ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], ms=4, lw=1.2, label=LABELS[m])
        draw_bands(ax, x, y,
                   z * sub[f"{metric}_noise_std"].to_numpy(),
                   z * sub[f"{metric}_test_std"].to_numpy(), COLORS[m])
    ax.set_xticks(list(eps_x.values()))
    ax.set_xticklabels([eps_tick(e) for e in eps_x])
    ax.invert_xaxis()
    ax.set_xlabel(r"Privacy level $\epsilon$")
    ax.set_ylabel("Negative flow [%]")
    ax.grid(True, alpha=0.3, ls="--")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + band_handles(z), l + [x.get_label() for x in band_handles(z)],
              loc="best", fontsize=9, frameon=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out_dir / f"negative_flow.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)

def print_negative_flow_table(agg, eps_list):
    metric = "negative_flow_pct"
    methods = [m for m in NEG_FLOW_METHODS if m in set(agg["method"])]

    if not methods:
        print("No negative_flow data found.")
        return

    eps_sorted = sorted(eps_list, reverse=True)

    # Header
    cols = "l" + "c" * len(eps_sorted)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Percentage of negative flows.}")
    print(f"\\begin{{tabular}}{{{cols}}}")
    print(r"\toprule")

    header = "Method & " + " & ".join(f"$\\epsilon={e}$" for e in eps_sorted) + r" \\"
    print(header)
    print(r"\midrule")

    for m in methods:
        row = [LABELS[m]]

        for eps in eps_sorted:
            sub = agg[(agg["method"] == m) & (agg["eps"] == eps)]

            if sub.empty or pd.isna(sub[f"{metric}_mean"].iloc[0]):
                row.append("--")
            else:
                mean = sub[f"{metric}_mean"].iloc[0]
                std = sub[f"{metric}_test_std"].iloc[0]
                row.append(f"{mean:.4f} $\\pm$ {std:.4f}")

        print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:negative_flow}")
    print(r"\end{table}")



# def plot_hourly(hourly_agg, method, out_dir, eps_list, z, target):
#     """One panel per eps, stacked vertically. Every panel has its own y-range and
#     its height is proportional to that range (same inches per % point).
#     The average bid volume is drawn on a right axis over the full panel height."""
#     sub = hourly_agg[hourly_agg["method"] == method]
#     eps_here = sorted([e for e in sub["eps"].unique()
#                        if (e==8.0 or e==2.0 or e==0.5) and (e in set(eps_list) or e >= NO_PRIVACY_EPS)], reverse=True)
#     if not eps_here:
#         return

#     BID_COLOR = "0.30"

#     # one common right-axis range for all panels of this figure
#     sub_e = sub[sub["eps"].isin(eps_here)]
#     bid_top = float(np.nanmax((sub_e["bid_mean"] + z * sub_e["bid_noise_std"].fillna(0)).to_numpy()))
#     bid_top = bid_top * 1.08 if bid_top > 0 else 1.0
#     print(f"Max bid volume: {bid_top:.2f} kW")

#     # ---- 1) data and independent y-range per panel ----
#     data, ylims = {}, {}
#     for e in eps_here:
#         d = sub[sub["eps"] == e].sort_values("hour")
#         y = d["success_mean"].to_numpy() * 100
#         noise = z * d["noise_std"].fillna(0).to_numpy() * 100
#         test = z * d["test_std"].fillna(0).to_numpy() * 100
#         by = d["bid_mean"].to_numpy()
#         bn = z * d["bid_noise_std"].fillna(0).to_numpy()
#         data[e] = (d["hour"].to_numpy(), y, noise, test, by, bn)
#         low = np.min(np.clip(y - np.maximum(noise, test), 0, 100))
#         ylims[e] = (min(58, max(0.0, np.floor((low - 3) / 5) * 5)), 101.0)

#     # ---- 2) layout in inches ----
#     in_per_pct = 0.024
#     h_min = 0.9
#     ax_w = 3.5
#     gap_in = 0.28
#     left_in, right_in = 0.55, 0.6      # room on the right for the bid axis
#     bottom_in = 0.45
#     top_in = 1.0                       # suptitle + legend + first panel title

#     heights = [max(h_min, in_per_pct * (ylims[e][1] - ylims[e][0])) for e in eps_here]

#     min_stack = 1.7
#     stack = sum(heights) + gap_in * (len(eps_here) - 1)
#     if stack < min_stack:
#         scale = (min_stack - gap_in * (len(eps_here) - 1)) / sum(heights)
#         heights = [h * scale for h in heights]
#         stack = min_stack
#     fig_w = ax_w + left_in + right_in
#     fig_h = sum(heights) + gap_in * (len(eps_here) - 1) + bottom_in + top_in
#     fig = plt.figure(figsize=(fig_w, fig_h))

#     color = COLORS[method]
#     axes, top = [], fig_h - top_in
#     for i, (e, hgt) in enumerate(zip(eps_here, heights)):
#         bottom = top - hgt
#         ax = fig.add_axes([left_in / fig_w, bottom / fig_h, ax_w / fig_w, hgt / fig_h],
#                           sharex=axes[0] if axes else None)
#         top = bottom - gap_in
#         axes.append(ax)

#         h, y, noise, test, by, bn = data[e]
#         ax.plot(h, y, color=color, lw=2)
#         ax.fill_between(h, np.clip(y - noise, 0, 100), np.clip(y + noise, 0, 100),
#                         color=color, alpha=0.22, lw=0)
#         ax.plot(h, np.clip(y + test, 0, 100), color=color, ls="--", lw=1.0)
#         ax.plot(h, np.clip(y - test, 0, 100), color=color, ls="--", lw=1.0)
#         ax.axhline(target * 100, color="red", lw=1, ls="--", alpha=1)
#         ax.set_ylim(*ylims[e])
#         ax.set_title("No privacy" if e >= NO_PRIVACY_EPS else rf"$\epsilon={e:g}$", pad=2)
#         ax.set_xticks(range(0, 24, 2))
#         ax.set_xticks(range(24), minor=True)
#         ax.tick_params(axis="both", pad=1.5)
#         ax.grid(True, which="major", alpha=0.3)
#         ax.grid(True, which="minor", alpha=0.12)
#         if i < len(eps_here) - 1:                 # x labels only on the bottom panel
#             ax.tick_params(axis="x", labelbottom=False)

#         # average bid volume (right axis), full panel height, may overlap success
#         ax2 = ax.twinx()
#         ax2.plot(h, by, color=BID_COLOR, ls="-.", lw=1.3)
#         ax2.fill_between(h, np.maximum(by - bn, 0), by + bn, color=BID_COLOR, alpha=0.15, lw=0)
#         ax2.set_ylim(0, max(bid_top*1.3,15000))
#         ax2.grid(True, which="major", alpha=0.3)
#         ax2.set_yticks(np.arange(0, max(bid_top*1.3,15000)+1, 5000))
#         #ax2.tick_params(axis="y", pad=1.5)
#         #ax.set_zorder(ax2.get_zorder() + 1)       # keep the success curve on top
#         #ax.patch.set_visible(False)

#     fig.supxlabel("Hour", y=0.0, va="bottom")
#     y_center = (bottom_in + stack / 2) / fig_h
#     fig.supylabel("Success [%]", x=0.0, y=y_center, ha="left", va="center")
#     fig.text(1.01 - 0.1 / fig_w, y_center, "Avg. bid volume [kW]", fontsize=11, fontname="Times New Roman",
#              rotation=270, va="center", ha="center")
#     fig.suptitle(LABELS[method], y=1.0, va="top", fontsize=11)
#     fig.legend(handles=band_handles(z) + [
#                    Line2D([0], [0], color="red", ls="--", lw=1, label=f"Target {target:.0%}"),
#                    Line2D([0], [0], color=BID_COLOR, ls="-.", lw=1.3,
#                           label="Avg. bid volume (right axis)"),
#                    Patch(facecolor=BID_COLOR, alpha=0.2, label="Bid volume: privacy-noise spread")],
#                loc="upper center", ncol=2,
#                bbox_to_anchor=(0.5, 1 - 0.26 / fig_h), frameon=False,
#                fontsize=8.5, handlelength=1.2, columnspacing=0.5, borderpad=0.1)

#     for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
#         fig.savefig(out_dir / f"hourly_success_{method}.{ext}",
#                     bbox_inches="tight", pad_inches=0.02, **kw)
#     plt.close(fig)


def plots_hourly(hourly_agg, method, out_dir, eps_list, z, target):
    """One panel per eps, stacked vertically. Every panel has its own y-range and
    its height is proportional to that range (same inches per % point)."""
    sub = hourly_agg[hourly_agg["method"] == method]
    eps_here = sorted([e for e in sub["eps"].unique()
                       if  (e in set(eps_list) or e >= NO_PRIVACY_EPS)], reverse=True) #or e == 2.0 #(e == 8.0  or e == 0.5 or e == NO_PRIVACY_EPS) and 
                       
    if not eps_here:
        return

    # ---- 1) data and independent y-range per panel ----
    data, ylims = {}, {}
    for e in eps_here:
        d = sub[sub["eps"] == e].sort_values("hour")
        y = d["success_mean"].to_numpy() * 100
        noise = z * d["noise_std"].fillna(0).to_numpy() * 100
        test = z * d["test_std"].fillna(0).to_numpy() * 100
        data[e] = (d["hour"].to_numpy(), y, noise, test)
        low = np.min(np.clip(y - np.maximum(noise, test), 0, 100))
        ylims[e] = (min(85,max(0.0, np.floor((low - 3) / 5) * 5)), 101.0)


    # ---- 2) layout in inches ----
    in_per_pct = 0.024            # panel height per % point (span 100 -> 2.4 in)
    h_min = 0.9                   # never smaller than this
    ax_w = 3.5                    # panel width
    gap_in = 0.28                 # space between panels (holds the panel title)
    left_in, right_in = 0.55, 0.1
    bottom_in = 0.45              # x tick labels + "Hour"
    top_in = 0.75                 # suptitle + legend + first panel title

    heights = [max(h_min, in_per_pct * (ylims[e][1] - ylims[e][0])) for e in eps_here]

    # NEW: the stack must be tall enough to hold the y label (~1.7 in)
    min_stack = 1.7
    stack = sum(heights) + gap_in * (len(eps_here) - 1)
    if stack < min_stack:
        scale = (min_stack - gap_in * (len(eps_here) - 1)) / sum(heights)
        heights = [h * scale for h in heights]
        stack = min_stack
    fig_w = ax_w + left_in + right_in
    fig_h = sum(heights) + gap_in * (len(eps_here) - 1) + bottom_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    color = COLORS[method]
    axes, top = [], fig_h - top_in
    for i, (e, hgt) in enumerate(zip(eps_here, heights)):
        bottom = top - hgt
        ax = fig.add_axes([left_in / fig_w, bottom / fig_h, ax_w / fig_w, hgt / fig_h],
                          sharex=axes[0] if axes else None)
        top = bottom - gap_in
        axes.append(ax)

        h, y, noise, test = data[e]
        ax.plot(h, y, color=color, lw=2)
        ax.fill_between(h, np.clip(y - noise, 0, 100), np.clip(y + noise, 0, 100),
                        color=color, alpha=0.22, lw=0)
        ax.plot(h, np.clip(y + test, 0, 100), color=color, ls="--", lw=1.0)
        ax.plot(h, np.clip(y - test, 0, 100), color=color, ls="--", lw=1.0)
        ax.axhline(target * 100, color="red", lw=1, ls="--", alpha=1)
        ax.set_ylim(*ylims[e])
        ax.set_title("No privacy" if e >= NO_PRIVACY_EPS else rf"$\epsilon={e:g}$", pad=2)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticks(range(24), minor=True)
        ax.tick_params(axis="both", pad=1.5)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        if i < len(eps_here) - 1:                 # x labels only on the bottom panel
            ax.tick_params(axis="x", labelbottom=False)

    fig.supxlabel("Hour", y=0.0, va="bottom")
    fig.supylabel("Success [%]", x=0.0, ha="left")
    fig.suptitle(LABELS[method], y=1.0, va="top", fontsize=11)
    fig.legend(handles=band_handles(z) + [Line2D([0], [0], color="red", ls="--", lw=1,
               label=f"Target {target:.0%}")],
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1 - 0.26 / fig_h), frameon=False,
               fontsize=11, handlelength=1.2, columnspacing=0.5, borderpad=0.1)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out_dir / f"hourly_success_{method}.{ext}",
                    bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)

def plots_hourly(hourly_agg, methods, out_dir, eps_list, z, target):
    """One panel per eps, stacked vertically. Every panel shows all methods in `methods`
    (each with its own colour). Every panel has its own y-range, fitted to all methods,
    and its height is proportional to that range (same inches per % point)."""
    if isinstance(methods, str):               # still works with a single method
        methods = [methods]
    sub = hourly_agg[hourly_agg["method"].isin(methods)]
    eps_here = sorted([e for e in sub["eps"].unique()
                       if (e in set(eps_list) or e >= NO_PRIVACY_EPS)], reverse=True)
    if not eps_here:
        return

    # ---- 1) data (per eps and method) and independent y-range per panel ----
    data, ylims = {}, {}
    for e in eps_here:
        lows = []
        for m in methods:
            d = sub[(sub["eps"] == e) & (sub["method"] == m)].sort_values("hour")
            if d.empty:
                continue
            y = d["success_mean"].to_numpy() * 100
            noise = z * d["noise_std"].fillna(0).to_numpy() * 100
            test = z * d["test_std"].fillna(0).to_numpy() * 100
            data[(e, m)] = (d["hour"].to_numpy(), y, noise, test)
            lows.append(np.min(np.clip(y - np.maximum(noise, test), 0, 100)))
        low = min(lows)                         # lowest point over all methods in this panel
        ylims[e] = (min(85, max(0.0, np.floor((low - 3) / 5) * 5)), 101.0)

    # ---- 2) layout in inches ----
    in_per_pct = 0.024            # panel height per % point (span 100 -> 2.4 in)
    h_min = 0.9                   # never smaller than this
    ax_w = 3.5                    # panel width
    gap_in = 0.28                 # space between panels (holds the panel title)
    left_in, right_in = 0.55, 0.1
    bottom_in = 0.45              # x tick labels + "Hour"
    top_in = 0.85                 # two-row legend + first panel title

    heights = [max(h_min, in_per_pct * (ylims[e][1] - ylims[e][0])) for e in eps_here]

    # the stack must be tall enough to hold the y label (~1.7 in)
    min_stack = 1.7
    stack = sum(heights) + gap_in * (len(eps_here) - 1)
    if stack < min_stack:
        scale = (min_stack - gap_in * (len(eps_here) - 1)) / sum(heights)
        heights = [h * scale for h in heights]
        stack = min_stack
    fig_w = ax_w + left_in + right_in
    fig_h = sum(heights) + gap_in * (len(eps_here) - 1) + bottom_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes, top = [], fig_h - top_in
    for i, (e, hgt) in enumerate(zip(eps_here, heights)):
        bottom = top - hgt
        ax = fig.add_axes([left_in / fig_w, bottom / fig_h, ax_w / fig_w, hgt / fig_h],
                          sharex=axes[0] if axes else None)
        top = bottom - gap_in
        axes.append(ax)

        for m in methods:
            if (e, m) not in data:
                continue
            color = COLORS[m]
            h, y, noise, test = data[(e, m)]
            ax.plot(h, y, color=color, lw=2)
            ax.fill_between(h, np.clip(y - noise, 0, 100), np.clip(y + noise, 0, 100),
                            color=color, alpha=0.22, lw=0)
            ax.plot(h, np.clip(y + test, 0, 100), color=color, ls="--", lw=1.0)
            ax.plot(h, np.clip(y - test, 0, 100), color=color, ls="--", lw=1.0)

        ax.axhline(target * 100, color="red", lw=1, ls="--", alpha=1)
        ax.set_ylim(*ylims[e])
        ax.set_title("No privacy" if e >= NO_PRIVACY_EPS else rf"$\epsilon={e:g}$", pad=2)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticks(range(24), minor=True)
        ax.tick_params(axis="both", pad=1.5)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        if i < len(eps_here) - 1:                 # x labels only on the bottom panel
            ax.tick_params(axis="x", labelbottom=False)

    y_center = (bottom_in + stack / 2) / fig_h
    fig.supxlabel("Hour", y=0.0, va="bottom")
    fig.supylabel("Success [%]", x=0.0, y=y_center, ha="left", va="center")

    # legend: one coloured line per method, then the band / target entries
    method_handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=LABELS[m])
                      for m in methods]
    fig.legend(handles=method_handles + band_handles(z) +
               [Line2D([0], [0], color="red", ls="--", lw=1, label=f"Target {target:.0%}")],
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1 - 0.02 / fig_h), frameon=False,
               fontsize=9.5, handlelength=1.2, columnspacing=0.8, borderpad=0.1)

    tag = "_&_".join(methods)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out_dir / f"hourly_success_{tag}.{ext}",
                    bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--eps", type=float, nargs="+", default=None,
                    help="epsilons to plot (default: all found below no-privacy)")
    ap.add_argument("--band-z", type=float, default=1.645,
                    help="band half-width in std units (1.645 ~ 90%% if normal)")
    ap.add_argument("--pct-oracle", action="store_true",
                    help="show objective and bid volume as %% of the oracle")
    ap.add_argument("--log", action="store_true",
                    help="log y-axis for objective / bid volume (only if all values > 0)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_dir / "seed_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run, hourly = collect(args.results_dir, args.target)
    agg = aggregate_scalars(per_run, args.target)
    hourly_agg = aggregate_hourly(hourly, args.target)

    per_run.to_csv(out_dir / "per_run_scalars.csv", index=False)
    hourly.to_csv(out_dir / "per_run_hourly.csv", index=False)
    agg.to_csv(out_dir / "summary_noise_vs_test.csv", index=False)
    hourly_agg.to_csv(out_dir / "hourly_noise_vs_test.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print("\nCell format: mean (±std from privacy noise / ±std from future days)")
    tables = [
        ("Objective value [EUR/day]", "objective_value", 1.0, "{:.0f}"),
        ("Mean bid success probability [%]", "empirical_feasibility", 100.0, "{:.1f}"),
        ("Worst-hour success probability [%]", "worst_hour_success", 100.0, "{:.1f}"),
        ("Hours below target [count]", "hours_below_target", 1.0, "{:.1f}"),
        ("Total bid volume [MW]", "total_bid_kw", 1.0, "{:.1f}"),
        ("Negative flow [% of 90*24]", "negative_flow_pct", 1.0, "{:.2f}"),
    ]
    for title, metric, scale, fmt in tables:
        if metric == "negative_flow_pct":
            a = agg[agg["method"].isin(NEG_FLOW_METHODS)]
            if a[f"{metric}_mean"].isna().all():
                continue
            t = pm_table(a, metric, scale, fmt)
        else:
            t = pm_table(agg, metric, scale, fmt)
        print(f"\n=== {title} ===")
        print(t.to_string())
        t.to_csv(out_dir / f"table_{metric}.csv")

    print("\n=== Which spread dominates? (mean success probability, in % points) ===")
    d = agg[["method", "eps"]].copy()
    d["noise_std"] = agg["empirical_feasibility_noise_std"] * 100
    d["test_std"] = agg["empirical_feasibility_test_std"] * 100
    d["noise/test"] = d["noise_std"] / d["test_std"]
    print(d.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nShare of runs meeting the {args.target:.0%} target in ALL hours [%]")
    share = agg.pivot(index="method", columns="eps", values="share_runs_all_hours_ok")
    share = share[sorted(share.columns, reverse=True)]
    share.columns = [eps_label(e) for e in share.columns]
    print((share * 100).round(0).to_string())

    if not args.no_plots:
        set_style()
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        eps_list = args.eps or sorted(
            e for e in agg["eps"].unique() if e < NO_PRIVACY_EPS)
        plot_comparison(agg, fig_dir, eps_list, args.band_z, args.pct_oracle, args.log)
        #plot_negative_flow(agg, fig_dir, eps_list, args.band_z)
        print_negative_flow_table(agg, eps_list)
        #for method in agg["method"].unique():
        #    plot_hourly(hourly_agg, method, fig_dir, eps_list, args.band_z, args.target)

        method = ["NPE", "Oracle"]
        plots_hourly(hourly_agg, method, fig_dir, eps_list, args.band_z, args.target)
        print(f"Saved figures to {fig_dir}")

    print(f"\nSaved all tables to {out_dir}")


if __name__ == "__main__":
    main()