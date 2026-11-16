#!/usr/bin/env python3
"""
Method CDPP90, eps = 0.5: which seeds have success < 0.9?

Input columns: seed,method,eps,hour,success

Reports
  1. seeds whose overall mean success is < threshold
  2. seeds with at least one hour below threshold (and which hours)
  3. a box plot of success per hour across seeds, with the values below the
     threshold shown as red dots labelled by seed

Usage:
    python cdpp90_low_seeds.py
    python cdpp90_low_seeds.py --input path/to/per_seed_hourly.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = "results/seed_analysis/per_run_hourly.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--method", default="CDP90")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--outdir", default=None, help="default: folder of the input file")
    args = ap.parse_args()

    path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    df = df[(df["method"] == args.method) & np.isclose(df["eps"], args.eps)]
    if df.empty:
        raise SystemExit(f"No rows for method={args.method}, eps={args.eps}")
    thr = args.threshold
    tag = f"{args.method}_eps_{args.eps}"

    # ---- 1. seeds whose overall mean success is below the threshold -----
    per_seed = df.groupby("seed")["success"].agg(mean="mean", min="min", n_hours="count")
    low_mean = per_seed[per_seed["mean"] < thr].sort_values("mean")
    print(f"{args.method}, eps={args.eps}: {df['seed'].nunique()} seeds")
    print(f"\nSeeds with overall mean success < {thr}:")
    print(low_mean.to_string() if not low_mean.empty else "  none")

    # ---- 2. seeds with at least one hour below the threshold ------------
    bad = df[df["success"] < thr]
    print(f"\nSeeds with at least one hour < {thr}:")
    if bad.empty:
        print("  none")
    else:
        by_seed = (
            bad.groupby("seed")
            .agg(n_bad_hours=("hour", "nunique"),
                 bad_hours=("hour", lambda s: sorted(s.unique().tolist())),
                 worst=("success", "min"))
            .sort_values("worst")
        )
        print(by_seed.to_string())
        by_seed.to_csv(outdir / f"low_seeds_{tag}.csv")

    # ---- 3. box plot per hour across seeds ------------------------------
    hours = sorted(df["hour"].unique())
    data = [df[df["hour"] == h]["success"].values for h in hours]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(hours)), 5))
    ax.boxplot(data, positions=range(len(hours)), widths=0.6, showfliers=False)
    for k, h in enumerate(hours):
        sub = df[(df["hour"] == h) & (df["success"] < thr)]
        ax.scatter([k] * len(sub), sub["success"], color="red", s=18, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(str(r["seed"]), (k, r["success"]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    ax.axhline(thr, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours)
    ax.set_xlabel("hour")
    ax.set_ylabel("success")
    ax.set_title(f"{args.method}, eps={args.eps}: success per hour across seeds "
                 f"(red = below {thr}, labelled by seed)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = outdir / f"boxplot_{tag}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved box plot to {out}")


if __name__ == "__main__":
    main()