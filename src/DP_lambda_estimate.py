"""
Infer EV arrival rates from records using local differential privacy.

This script:

1. Loads the EV records and state definitions.
2. Estimates state-dependent arrival rates using LDP.
3. Estimates the corresponding non-private arrival rates.
4. Saves both results and metadata to a pickle file.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import lam_price_setting as lps


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Infer EV arrival rates using local differential privacy."
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time discretization step in hours.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=5.0,
        help="LDP privacy budget.",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=24,
        help="Number of time bins used in the output filename.",
    )

    parser.add_argument(
        "--p-rate",
        type=int,
        default=12,
        help="Assumed EV charging power in kW.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Arrival-rate scaling factor.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output pickle path.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")

    if args.eps < 0:
        raise ValueError("--eps must be non-negative.")

    if args.H <= 0:
        raise ValueError("--H must be positive.")

    if args.p_rate <= 0:
        raise ValueError("--p-rate must be positive.")

    if args.scale <= 0:
        raise ValueError("--scale must be positive.")


def default_output_path(args: argparse.Namespace) -> Path:
    """Construct the default output path."""
    return Path(
        f"data/lambda_inference_"
        f"{args.dt:.2f}_{args.eps}_{args.H}_{args.scale}.pkl"
    )


def run_inference(args: argparse.Namespace) -> dict:
    """
    Run private and non-private arrival-rate estimation.

    Returns
    -------
    dict
        Dictionary containing estimated rates and metadata.
    """
    n_time_steps = int(round(24 / args.dt))

    print(
        f"Parameters: dt={args.dt}, eps={args.eps}, "
        f"H={args.H}, p_rate={args.p_rate}, scale={args.scale}"
    )

    print("Loading EV records and state definitions...")
    _, status = lps.load_ev_model_parameters(
        args.dt,
        args.p_rate,
        args.scale,
    )

    print("Inferring privatized arrival rates...")

    lambda_private, status, sigma, covariance = (
        lps.estimate_arrival_rate_from_records(
            states=status,
            n_time_steps=n_time_steps,
            time_step_h=args.dt,
            n_chargers=args.scale,
            privacy_mechanism="LDP",
            epsilon=args.eps,
        )
    )

    print("Inferring non-private arrival rates...")
    lambda_non_private, _ = (
        lps.estimate_arrival_rate_from_records(
            states=status,
            n_time_steps=n_time_steps,
            time_step_h=args.dt,
            n_chargers=args.scale,
            privacy_mechanism=None,
        )
    )

    return {
        "results": {
            "time_bin": np.arange(n_time_steps).tolist(),
            "lambda_hat": np.asarray(lambda_private).tolist(),
            "lambda_real": np.asarray(lambda_non_private).tolist(),
        },
        "metadata": {
            "dt": args.dt,
            "epsilon": args.eps,
            "n_time_steps": n_time_steps,
            "charging_power_kw": args.p_rate,
            "scale": args.scale,
            "sigma": np.asarray(sigma).tolist(),
            "Sigma_lambda": np.asarray(covariance).tolist(),
            "status": np.asarray(status).tolist(),
        },
    }


def main() -> None:
    """Run the inference pipeline and save the results."""
    args = parse_arguments()
    validate_arguments(args)

    output_path = args.output or default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_inference(args)

    with output_path.open("wb") as file:
        pickle.dump(results, file)

    print(f"Saved inference results to: {output_path}")


if __name__ == "__main__":
    main()