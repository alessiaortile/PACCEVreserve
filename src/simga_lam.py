"""
Hourly relative estimation error sigma_hat / lambda_hat for LDP and CDP.

For each mechanism, epsilon and seed, the estimator is run through
lam_price_setting.estimate_arrival_rate_from_records. The hourly ratio is

    r_t = sqrt( sum_x Var(lambda_hat_{x,t}) ) / sum_x lambda_hat_{x,t}

(aggregated over states to avoid divisions by lambda_hat = 0), averaged over
seeds and written as a LaTeX table (requires booktabs and graphicx).
"""
import argparse
from pathlib import Path

import numpy as np

import lam_price_setting as lps

def flexible_mask(states):
    """True for states with tau_r > 0 and tau_s > 0 (same selector as W in the paper)."""
    return np.array([(c > 0 and s > 0) for c, s in states], dtype=bool)

def hourly_ratio(lambda_hat, var_hat, mask):
    """
    lambda_hat: (n_states, T); var_hat: (T, n_states); mask: (n_states,)
    Ratio computed over flexible states only:
        sqrt(sum_{x in F} Var_x) / sum_{x in F} lambda_hat_x
    """
    sigma = np.sqrt(var_hat[:, mask].sum(axis=1))      # (T,)
    total_rate = lambda_hat[mask, :].sum(axis=0)       # (T,)
    return sigma / np.maximum(total_rate, 1e-12)

def ratio_for(mechanism, eps, seed, states, n_time_steps, dt, scale, mask):
    out = lps.estimate_arrival_rate_from_records(
        states=states,
        n_time_steps=n_time_steps,
        time_step_h=dt,
        n_chargers=scale,
        privacy_mechanism=mechanism,
        epsilon=eps,
        seed=seed,
    )
    lambda_hat = out[0]

    if mechanism == "LDP":
        # returns (lambda_hat, states, Sigma_tilde_a, Sigma_lambda, proj_err)
        Sigma_lambda = out[3]
    else:
        # CDP returns (lambda_hat, states, Sigma_tilde_a, Sigma_poisson, proj_err):
        # the 4th output is the Poisson part, so recover Sigma_lambda.
        Sigma_lambda = [St - Sp for St, Sp in zip(out[2], out[3])]

    var_hat = np.array([np.diag(S) for S in Sigma_lambda])  # (T, n_states)
    return hourly_ratio(lambda_hat, var_hat, mask)


def fmt(v, bold_above):
    s = f"{v:.2f}"
    return rf"\textbf{{{s}}}" if v >= bold_above else s


def build_table(results, eps_list, n_time_steps, bold_above):
    cols = "l" + "r" * n_time_steps
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Hourly relative estimation error $\hat\sigma_t/\hat\lambda_t$ "
        r"(see text) for LDP and CDP, averaged over privacy-noise seeds.}",
        r"\label{tab:rel_error}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        r"$\varepsilon$ & " + " & ".join(str(h) for h in range(n_time_steps)) + r" \\",
    ]
    for mech in ("LDP", "CDP"):
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{n_time_steps + 1}}}{{l}}{{\textit{{{mech}}}}} \\"
        )
        for eps in eps_list:
            row = results[mech][eps]
            lines.append(
                f"{eps:g} & " + " & ".join(fmt(v, bold_above) for v in row) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="+", default=[8, 4, 2, 1, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 51)))
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--prate", type=int, default=12)
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--bold-above", type=float, default=1.0,
                    help="Ratios >= this value are printed in bold.")
    ap.add_argument("--out", type=Path, default=Path("rel_error_table.tex"))
    args = ap.parse_args()

    args.seeds = [1043, 1044, 1045, 1046, 1222, 1224, 1226, 1228, 1230, 1235, 1240, 1500, 1550, 1600, 1650, 3157, 3333, 3444, 3465, 3581, 3651, 3747, 3846, 3855, 3912, 4012, 4038, 4120, 4238, 4327, 4820,
    4873, 4876, 5000, 5100, 5150, 6122, 6450, 6710, 6825, 6840, 7834, 9650, 36511, 36522, 47336, 47895, 51234] #63205, 71785, 627282

    n_time_steps = int(round(args.hours / args.dt))
    lam, states = lps.load_ev_model_parameters(
        time_step_h=args.dt, charging_power_kw=args.prate, scale=args.scale
    )

    mask = flexible_mask(states)

    results = {"LDP": {}, "CDP": {}}
    for mech in ("LDP", "CDP"):
        for eps in args.eps:
            eps = float(eps)  # keeps "8.0" in the LDP file names
            per_seed = [
                ratio_for(mech, eps, s, states, n_time_steps, args.dt, args.scale, mask)
                for s in args.seeds
            ]
            results[mech][eps] = np.mean(per_seed, axis=0)

    tex = build_table(results, [float(e) for e in args.eps], n_time_steps, args.bold_above)
    args.out.write_text(tex)
    print(tex)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()