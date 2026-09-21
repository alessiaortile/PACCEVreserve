"""
Utilities for processing EV charging-session data.

This module:
1. Loads and cleans the raw EV charging dataset.
2. Discretizes charging sessions into the EV state space.
3. Estimates the time-dependent arrival rate lambda_t.
4. Estimates the conditional distribution of charging and slack times
   given the arrival time.
5. Exports the resulting model parameters to JSON.

The resulting parameters are used as inputs to the stochastic EV
population model described in the paper.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_and_clean_ev_data(input_path: str | Path) -> pd.DataFrame:
    """
    Load and clean the raw EV charging-session dataset.

    Parameters
    ----------
    input_path : str or Path
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned charging-session data.

    Notes
    -----
    The raw dataset is expected to contain:
        - connectionTime_decimal
        - chargingDuration
        - kWhDelivered
        - dayIndicator

    These columns are renamed to match the notation used throughout
    the project.
    """
    df = pd.read_csv(input_path)

    # Rename raw dataset columns to project-wide names.
    df = df.rename(
        columns={
            "connectionTime_decimal": "arrival_time",
            "chargingDuration": "duration_h",
        }
    )

    # Remove sessions with missing values in the variables required
    # for the EV state representation.
    df = df.dropna(
        subset=["arrival_time","duration_h","kWhDelivered","dayIndicator"]
    ).copy()

    # Keep physically meaningful sessions within the expected ranges.
    df = df[
        (df["arrival_time"] >= 0) & (df["arrival_time"] <= 24)
        & (df["duration_h"] > 0) & (df["duration_h"] < 24)
        & (df["kWhDelivered"] > 0) & (df["kWhDelivered"] < 150)
    ].copy()

    return df


def compute_ev_state_variables(
    df: pd.DataFrame,
    time_step_h: float,
    charging_rate_kw: float,
) -> pd.DataFrame:
    """
    Compute discretized EV charging and slack states.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned charging-session data.
    time_step_h : float
        Duration of one discretization interval in hours.
    charging_rate_kw : float
        Assumed common EV charging power in kW.

    Returns
    -------
    pd.DataFrame
        Dataframe containing charging time, slack time, and their
        corresponding discrete state indices.

    """
    if time_step_h <= 0:
        raise ValueError("time_step_h must be positive.")

    if charging_rate_kw <= 0:
        raise ValueError("charging_rate_kw must be positive.")

    df = df.copy()

    # Continuous-time EV characteristics.
    df["charging_time_h"] = df["kWhDelivered"] / charging_rate_kw
    df["slack_h"] = df["duration_h"] - df["charging_time_h"]

    # Remove sessions that cannot satisfy their charging requirement
    # at the assumed charging rate.
    df = df[df["slack_h"] >= 0].copy()

    # Discretize arrival time, charging time, and slack time.
    df["arrival_bin"] = (df["arrival_time"] / time_step_h).astype(int)
    df["charge_bin"] = (df["charging_time_h"] / time_step_h).astype(int)
    df["slack_bin"] = (df["slack_h"] / time_step_h).astype(int)

    return df


def estimate_arrival_rate(
    df: pd.DataFrame,
) -> dict[int, float]:
    """
    Estimate the time-dependent EV arrival rate.

    Parameters
    ----------
    df : pd.DataFrame
        EV charging-session data containing ``dayIndicator`` and
        ``arrival_bin``.

    Returns
    -------
    dict[int, float]
        Estimated mean number of arrivals for each arrival-time bin.

    """
    arrivals_per_day = (
        df.groupby(["dayIndicator", "arrival_bin"])
        .size()
        .reset_index(name="arrival_count")
    )

    arrival_rate = (
        arrivals_per_day.groupby("arrival_bin")["arrival_count"]
        .mean()
    )

    return {
        int(arrival_bin): float(rate)
        for arrival_bin, rate in arrival_rate.items()
    }


def estimate_conditional_state_distribution(
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Estimate the conditional EV state distribution given arrival time.

    Parameters
    ----------
    df : pd.DataFrame
        EV charging-session data containing ``arrival_bin``,
        ``charge_bin``, and ``slack_bin``.

    Returns
    -------
    dict
        Nested dictionary containing

            P(charge_bin, slack_bin | arrival_bin).

        The dictionary format is chosen so that the result can be
        serialized directly to JSON.
    """
    state_distribution = {}

    for arrival_bin in sorted(df["arrival_bin"].unique()):
        arrivals_at_bin = df[df["arrival_bin"] == arrival_bin]

        joint_counts = (
            arrivals_at_bin
            .groupby(["charge_bin", "slack_bin"])
            .size()
        )

        probabilities = joint_counts / joint_counts.sum()

        state_distribution[str(arrival_bin)] = {
            f"{charge_bin},{slack_bin}": round(float(probability), 4)
            for (charge_bin, slack_bin), probability in probabilities.items()
        }

    return state_distribution


def compute_ev_model_parameters(
    df: pd.DataFrame,
    time_step_h: float,
    charging_rate_kw: float,
) -> dict:
    """
    Compute all stochastic EV model parameters from historical data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned EV charging-session data.
    time_step_h : float
        Duration of one discretization interval in hours.
    charging_rate_kw : float
        Assumed common EV charging power in kW.

    Returns
    -------
    dict
        Model parameters containing:
            - discretization step
            - time-dependent arrival rates
            - conditional state distributions
    """
    df = compute_ev_state_variables(
        df=df,
        time_step_h=time_step_h,
        charging_rate_kw=charging_rate_kw,
    )

    arrival_rate = estimate_arrival_rate(df)

    state_distribution = estimate_conditional_state_distribution(df)

    return {
        "delta_t_hours": time_step_h,
        "lambda_t": arrival_rate,
        "conditional_distribution": state_distribution,
    }


def save_model_parameters(
    model_parameters: dict,
    args: argparse.Namespace,
) -> None:
    """
    Save estimated EV model parameters to a JSON file.

    Parameters
    ----------
    model_parameters : dict
        Dictionary returned by ``compute_ev_model_parameters``.
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    output_path = Path(f"{args.output}_{args.dt:.2f}_{args.charging_rate}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(model_parameters, file, indent=4)

    print(f"Model parameters saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate stochastic EV population model parameters."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/SYNTHETIC_EV_DATA.csv",
        help="Path to the input EV charging-session CSV file.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time discretization step in hours.",
    )

    parser.add_argument(
        "--charging_rate",
        type=float,
        default=12,
        help="Assumed EV charging power in kW.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/ev_model_parameters",
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the EV data-processing pipeline."""
    args = parse_arguments()

    df = load_and_clean_ev_data(args.input)

    model_parameters = compute_ev_model_parameters(
        df=df,
        time_step_h=args.dt,
        charging_rate_kw=args.charging_rate,
    )

    save_model_parameters(
        model_parameters=model_parameters,
        args=args,
    )


if __name__ == "__main__":
    main()

