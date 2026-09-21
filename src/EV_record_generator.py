"""
Generate synthetic electric-vehicle arrival records.

This module uses the EV model parameters loaded by ``lam_price_setting``.
For each simulated day, arrivals are sampled from independent Poisson
distributions:

    A[i, t] ~ Poisson(lambda[i, t])

where ``i`` denotes an EV state and ``t`` denotes a time interval.

Two record-generation modes are supported:

1. Standard generation:
   The sampled EV state is retained.

2. Local differential privacy (LDP):
   The sampled state is privatized using K-ary randomized response.

The generated records can be saved as one CSV file or split into
in-sample and out-of-sample datasets by day.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

import lam_price_setting as lps


State = Tuple[int, int]


def record_columns() -> List[str]:
    """Return the column names used in generated EV-record dataframes."""
    return [
        "day",
        "time_step",
        "state_idx",
        "status_charge",
        "status_slack",
        "arrival_time_hours",
        "slack_time_hours",
        "energy_charging_time_hours",
        "event_id",
    ]


def empty_records_df() -> pd.DataFrame:
    """"Return an empty dataframe with the standard EV-record schema."""
    return pd.DataFrame(columns=record_columns())

def _event_record(
    day: int,
    time_step: int,
    state_idx: int,
    charge: int,
    slack: int,
    arrival_time: float,
    dt: float,
    event_id: str,
) -> dict:
    """Create a dictionary representing one simulated EV arrival."""
    return {
        "day": day,
        "time_step": time_step,
        "state_idx": state_idx,
        "status_charge": charge,
        "status_slack": slack,
        "arrival_time_hours": arrival_time,
        "slack_time_hours": float(slack * dt),
        "energy_charging_time_hours": float(charge * dt),
        "event_id": event_id,
    }


def _sort_records(records: list[dict]) -> pd.DataFrame:
    """Convert records to a dataframe sorted by day and arrival time."""
    if not records:
        return empty_records_df()

    return (
        pd.DataFrame.from_records(records, columns=record_columns())
        .sort_values(
            ["day", "arrival_time_hours", "time_step", "state_idx"]
        )
        .reset_index(drop=True)
    )


def build_event_records(
    lam: np.ndarray,
    status: Sequence[State],
    days: int,
    dt: float,
    seed: int,
    hours: float,
) -> pd.DataFrame:
    """
    Generate non-private synthetic EV arrival records.

    Parameters
    ----------
    lam:
        State-dependent Poisson arrival rates with shape
        ``(n_states, n_time_steps)``.
    status:
        EV states represented as ``(charging_steps, slack_steps)``.
    days:
        Number of days to simulate.
    dt:
        Duration of one time step in hours.
    seed:
        Seed for reproducible random sampling.
    hours:
        Duration of one simulated day in hours.

    Returns
    -------
    pd.DataFrame
        Generated EV records using the true state labels.
    """

    rng = np.random.default_rng(seed)
    n_states, n_time_steps = lam.shape
    records: list[dict] = []

    for day in range(days):
        arrivals = rng.poisson(lam)

        for time_step in range(n_time_steps):
            step_start = time_step * dt

            for state_idx in range(n_states):
                count = int(arrivals[state_idx, time_step])
                if count == 0:
                    continue

                charge, slack = status[state_idx]

                for event_number in range(count):
                    arrival_time = (
                        day * hours
                        + step_start
                        + float(rng.uniform(0.0, dt))
                    )

                    records.append(
                        _event_record(
                            day=day,
                            time_step=time_step,
                            state_idx=state_idx,
                            charge=charge,
                            slack=slack,
                            arrival_time=arrival_time,
                            dt=dt,
                            event_id=(
                                f"d{day}_t{time_step}_s"
                                f"{state_idx}_{event_number}"
                            ),
                        )
                    )

    return _sort_records(records)


def build_DPevent_records(
    lam: np.ndarray,
    status: Sequence[State],
    days: int,
    dt: float,
    seed: int,
    hours: float,
    eps: float = 5.0,
) -> pd.DataFrame:
    """
    Generate EV records with K-ary randomized response.

    Arrival counts are sampled from the true Poisson rates, then each
    EV state is independently privatized using the specified privacy
    parameter.

    Parameters
    ----------
    lam, status, days, dt, seed, hours:
        Same as :func:`build_event_records`.
    status:
        EV states represented as ``(charging_steps, slack_steps)``.
    days:
        Number of days to simulate.
    dt:
        Duration of one time step in hours.
    seed:
        Seed for reproducible random sampling.
    hours:
        Duration of one simulated day in hours.
    eps:
        Non-negative LDP privacy budget. Smaller values provide stronger
        privacy and produce more state randomization.

    Returns
    -------
    pd.DataFrame
        Generated EV records with privatized state labels.
    """

    if eps < 0:
        raise ValueError("eps must be non-negative.")

    rng = np.random.default_rng(seed)
    n_states, n_time_steps = lam.shape
    records: list[dict] = []

    rr_matrix = lps.build_rr_matrix(
        epsilon=eps,
        n_states=n_states,
    )

    for day in range(days):
        arrivals = rng.poisson(lam)

        for time_step in range(n_time_steps):
            step_start = time_step * dt

            for true_state_idx in range(n_states):
                count = int(arrivals[true_state_idx, time_step])
                if count == 0:
                    continue

                for event_number in range(count):
                    private_state_idx = int(
                        rng.choice(
                            n_states,
                            p=rr_matrix[:, true_state_idx],
                        )
                    )

                    charge, slack = status[private_state_idx]

                    arrival_time = (
                        day * hours
                        + step_start
                        + float(rng.uniform(0.0, dt))
                    )

                    records.append(
                        _event_record(
                            day=day,
                            time_step=time_step,
                            state_idx=private_state_idx,
                            charge=charge,
                            slack=slack,
                            arrival_time=arrival_time,
                            dt=dt,
                            event_id=(
                                f"d{day}_t{time_step}_s"
                                f"{true_state_idx}_rr"
                                f"{private_state_idx}_{event_number}"
                            ),
                        )
                    )

    return _sort_records(records)


def split_records_by_day(
    df: pd.DataFrame,
    insample_days: int,
    outsample_days: int,
    seed: int = 1242,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split EV records into in-sample and out-of-sample datasets.

    Parameters
    ----------
    df:
        EV-record dataframe containing a ``day`` column.
    insample_days:
        Number of days assigned to the in-sample dataset.
    outsample_days:
        Number of days assigned to the out-of-sample dataset.
    seed:
        Seed used to shuffle the available days.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        In-sample and out-of-sample record dataframes.
    """
    if insample_days < 0:
        raise ValueError("insample_days must be non-negative.")

    if outsample_days < 0:
        raise ValueError("outsample_days must be non-negative.")

    if "day" not in df.columns:
        raise ValueError("The dataframe must contain a 'day' column.")

    available_days = np.sort(df["day"].unique())
    rng = np.random.default_rng(seed)
    shuffled_days = rng.permutation(available_days)

    insample_set = set(shuffled_days[:insample_days])
    oos_start = insample_days
    oos_end = oos_start + outsample_days
    oos_set = set(shuffled_days[oos_start:oos_end])

    insample = (
        df[df["day"].isin(insample_set)]
        .copy()
        .reset_index(drop=True)
    )

    outsample = (
        df[df["day"].isin(oos_set)]
        .copy()
        .reset_index(drop=True)
    )

    return insample, outsample


def build_split_output_path(base_output: Path, suffix: str) -> Path:
    """Return an output path with ``suffix`` added to its filename stem."""
    return base_output.with_name(
        f"{base_output.stem}_{suffix}{base_output.suffix}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic EV arrival records."
    )

    parser.add_argument(
        "--days",
        type=int,
        default=153,
        help="Number of days to simulate.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Length of one time step in hours.",
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Length of one simulated day in hours.",
    )

    parser.add_argument(
        "--prate",
        type=int,
        default=12,
        help="Charging rate used to load the EV model parameters.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1042,
        help="Random seed.",
    )

    parser.add_argument(
        "--DP",
        type=str,
        default=None,
        choices=["LDP"],
        help="Apply local differential privacy using randomized response.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=5.0,
        help="LDP privacy parameter.",
    )

    parser.add_argument(
        "--insample-days",
        type=int,
        default=60,
        help="Number of days assigned to the in-sample dataset.",
    )

    parser.add_argument(
        "--oos-days",
        type=int,
        default=30,
        help="Number of days assigned to the out-of-sample dataset.",
    )

    parser.add_argument(
        "--insample-output",
        type=Path,
        default=None,
        help="Optional output path for the in-sample CSV.",
    )

    parser.add_argument(
        "--oos-output",
        type=Path,
        default=None,
        help="Optional output path for the out-of-sample CSV.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Scale factor applied to the arrival rates.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ev_records.csv"),
        help="Base output CSV path.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the EV record-generation pipeline."""
    args = parse_args()

    if args.days <= 0:
        raise ValueError("days must be positive.")

    if args.insample_days > args.days:
        raise ValueError(
            "insample-days cannot exceed the total number of days."
        )

    if args.oos_days > args.days - args.insample_days:
        raise ValueError(
            "oos-days cannot exceed the remaining number of days."
        )

    if args.scale <= 0:
        raise ValueError("scale must be positive.")

    n_time_steps = int(round(args.hours / args.dt))

    lam, status = lps.load_ev_model_parameters(
        time_step_h=args.dt,
        charging_power_kw=args.prate,
        scale=args.scale,
    )

    if lam.shape[1] < n_time_steps:
        raise ValueError(
            "The loaded arrival-rate model contains fewer time steps "
            "than requested."
        )

    lam = lam[:, :n_time_steps]

    if args.DP == "LDP":
        records = build_DPevent_records(
            lam=lam,
            status=status,
            days=args.days,
            dt=args.dt,
            seed=args.seed,
            hours=args.hours,
            eps=args.eps,
        )

        args.output = args.output.with_name(
            f"{args.output.stem}_{args.scale}"
            f"_DP_eps{args.eps}{args.output.suffix}"
        )
    else:
        records = build_event_records(
            lam=lam,
            status=status,
            days=args.days,
            dt=args.dt,
            seed=args.seed,
            hours=args.hours,
        )

        args.output = args.output.with_name(
            f"{args.output.stem}_{args.scale}{args.output.suffix}"
        )

    insample, outsample = split_records_by_day(
        df=records,
        insample_days=args.insample_days,
        outsample_days=args.oos_days,
        seed=args.seed,
    )

    insample_output = (
        args.insample_output
        or build_split_output_path(args.output, "insample")
    )

    oos_output = (
        args.oos_output
        or build_split_output_path(args.output, "oos")
    )

    insample_output.parent.mkdir(parents=True, exist_ok=True)
    oos_output.parent.mkdir(parents=True, exist_ok=True)

    insample.to_csv(insample_output, index=False)
    outsample.to_csv(oos_output, index=False)

    print(f"Generated {len(insample)} in-sample records.")
    print(f"Generated {len(outsample)} out-of-sample records.")
    print(f"Saved in-sample data to: {insample_output}")
    print(f"Saved out-of-sample data to: {oos_output}")
    print(
        f"Configuration: dt={args.dt}, hours={args.hours}, "
        f"scale={args.scale}, seed={args.seed}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())