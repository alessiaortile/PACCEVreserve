"""
Arrival-rate estimation and privacy-aware uncertainty modelling.

This module provides utilities for:

1. Loading EV arrival-rate and conditional-state-distribution parameters.
2. Constructing the discretized EV state space.
3. Estimating true arrival rates from LDP-privatized observations
   using non-negative generalized least squares.
4. Constructing predictive arrival covariance matrices.
5. Constructing the K-ary randomized-response matrix used by LDP.

Notation
--------
The implementation follows the notation used in the paper:

    lambda_t
        Total expected number of EV arrivals at time t.

    lambda_x,t
        Expected number of arrivals in EV state x at time t.

    f_x,t
        Conditional probability of an EV being in state x given
        arrival at time t.

Thus,

    lambda_x,t = lambda_t * f_x,t.

For the LDP model, the randomized-response mechanism is represented
by a transition matrix H.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, nnls


# ============================================================================
# Market-data utilities
# ============================================================================

def read_daily_prices(
    fcrd_file: str,
    spot_file: str,
) -> dict:
    """
    Load and align hourly FCR-D and spot-market prices.

    Parameters
    ----------
    fcrd_file : str
        Path to the FCR-D market data.
    spot_file : str
        Path to the spot-market data.

    Returns
    -------
    dict
        Dictionary indexed by day. Each entry contains timestamps,
        FCR-D up/down prices, and spot prices.

    Notes
    -----
    Prices are divided by 1000 to convert the input units to the
    units used by the optimization model.
    """
    # ------------------------------------------------------------------
    # FCR-D data
    # ------------------------------------------------------------------
    fcrd = pd.read_csv(fcrd_file, sep=";")

    fcrd.columns = [
        "start_time",
        "end_time",
        "fcrd_down",
        "fcrd_up",
    ]

    fcrd["start_time"] = pd.to_datetime(
        fcrd["start_time"],
        utc=True,
    )

    fcrd = fcrd[
        ["start_time", "fcrd_down", "fcrd_up"]
    ]

    # ------------------------------------------------------------------
    # Spot-market data
    # ------------------------------------------------------------------
    spot = pd.read_csv(spot_file)

    spot["utc_time"] = pd.to_datetime(
        spot["utc_time"],
        utc=True,
    )

    spot = spot.rename(
        columns={
            "utc_time": "start_time",
            "price_mwh": "spot_price",
        }
    )

    # ------------------------------------------------------------------
    # Merge and verify timestamp alignment
    # ------------------------------------------------------------------
    merged = pd.merge(
        fcrd,
        spot,
        on="start_time",
        how="inner",
    )

    if len(merged) != len(fcrd):
        raise ValueError(
            "Timestamp mismatch between FCR-D and spot data. "
            f"FCR-D rows={len(fcrd)}, "
            f"matched rows={len(merged)}."
        )

    merged["day"] = merged["start_time"].dt.date

    daily_data = {}

    for day_index, (_, day_data) in enumerate(
        merged.groupby("day"),
    ):
        day_data = day_data.sort_values("start_time")

        daily_data[day_index] = {
            "timestamps": day_data["start_time"].to_numpy(),
            "fcrd_down": day_data["fcrd_down"].to_numpy() / 1000,
            "fcrd_up": day_data["fcrd_up"].to_numpy() / 1000,
            "spot": day_data["spot_price"].to_numpy() / 1000,
        }

    return daily_data


# ============================================================================
# Numerical utilities
# ============================================================================

def nnls_stable(
    matrix: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Solve a non-negative least-squares problem with a bounded fallback.

    Parameters
    ----------
    matrix:
        Regression or observation matrix.
    observations:
        Observed response vector.

    Returns
    -------
    np.ndarray
        Non-negative least-squares solution.
    """
    result = lsq_linear(
        matrix,
        observations,
        bounds=(-np.inf, np.inf), #0 for lower
        method="trf",
        lsmr_tol="auto",
    )

    return np.clip(result.x, 1e-8, None)


# ============================================================================
# EV model parameter loading
# ============================================================================

def load_ev_model_parameters(
    time_step_h: float = 1 / 3,
    charging_power_kw: float = 12,
    scale: int = 100,
) -> Tuple[np.ndarray, list[Tuple[int, int]]]:
    """
    Load EV model parameters and construct state-dependent arrival rates.

    Parameters
    ----------
    time_step_h:
        Duration of one EV-model time step in hours.
    charging_power_kw:
        Assumed EV charging power in kW.
    scale:
        Factor used to scale the empirical arrival rates.

    Returns
    -------
    tuple
        State-dependent arrival rates with shape ``(n_states, T)``
        and the corresponding EV state list.
    """
    parameter_path = (
        Path("data")
        / f"ev_model_parameters_{time_step_h:.2f}_{charging_power_kw}.json"
    )

    print(
        f"Loading EV model parameters: "
        f"dt={time_step_h:.2f} h, "
        f"charging power={charging_power_kw} kW, "
        f"scale={scale}"
    )

    with parameter_path.open("r") as file:
        data = json.load(file)

    time_step_h = data["delta_t_hours"]
    total_arrival_rate_dict = data["lambda_t"]
    conditional_distribution = data["conditional_distribution"]

    # ------------------------------------------------------------------
    # Total arrival rate lambda_t
    # ------------------------------------------------------------------
    time_bins = sorted(
        int(time_bin)
        for time_bin in total_arrival_rate_dict
    )

    total_arrival_rate = np.array(
        [
            total_arrival_rate_dict[str(time_bin)]
            for time_bin in time_bins
        ],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Construct the complete state space
    # ------------------------------------------------------------------
    states = set()

    for distribution in conditional_distribution.values():
        for state_string in distribution:
            charge, slack = map(
                int,
                state_string.split(","),
            )
            states.add((charge, slack))

    # The optimization model uses the complete rectangular state space.
    max_charge = max(charge for charge, _ in states)
    max_slack = max(slack for _, slack in states)

    states = {
        (charge, slack)
        for charge in range(max_charge + 1)
        for slack in range(max_slack + 1)
    }

    states = sorted(states)

    # ------------------------------------------------------------------
    # State-dependent arrival rate lambda_x,t
    # ------------------------------------------------------------------
    n_time_steps = len(time_bins)
    n_states = len(states)

    arrival_rate = np.zeros(
        (n_states, n_time_steps),
        dtype=float,
    )

    for time_index, time_bin in enumerate(time_bins):

        distribution = conditional_distribution.get(
            str(time_bin),
            {},
        )

        for state_index, state in enumerate(states):
            probability = distribution.get(
                f"{state[0]},{state[1]}",
                0.0,
            )

            arrival_rate[state_index, time_index] = (
                scale
                * total_arrival_rate[time_index]
                * probability
            )

    return arrival_rate, states


# ============================================================================
# LDP: randomized-response estimator
# ============================================================================

def estimate_arrival_rate_ldp(
    rr_matrix: np.ndarray,
    privatized_arrival_rate: np.ndarray,
    states: list[Tuple[int, int]],
    epsilon: float,
    n_days: int = 1,
    max_iter: int = 50,
    tolerance: float = 1e-6,
    regularization: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate true arrival rates from randomized-response observations.

    A non-negative iteratively reweighted generalized least-squares
    problem is solved independently for each time step.

    Parameters
    ----------
    rr_matrix:
        Randomized-response transition matrix.
    privatized_arrival_rate:
        Observed privatized rates with shape ``(n_states, T)``.
    n_days:
        Number of independent observation days.
    max_iter:
        Maximum number of reweighting iterations.
    tolerance:
        Relative convergence tolerance.
    regularization:
        Small numerical stabilization constant.

    Returns
    -------
    tuple
        Estimated arrival rates and their covariance matrices.
    """
    rr_matrix = np.asarray(rr_matrix,dtype=float)

    privatized_arrival_rate = np.asarray(privatized_arrival_rate,dtype=float)

    n_states, n_time_steps = privatized_arrival_rate.shape

    lambda_hat = np.zeros((n_states, n_time_steps),dtype=float)

    covariance = np.zeros((n_time_steps, n_states, n_states),dtype=float)
    print(f"Conditoning rr_matrix for epsilon={epsilon:.2f}: {np.linalg.cond(rr_matrix):.2e}, ")


    for time_index in range(n_time_steps):

        observations = privatized_arrival_rate[:,time_index]

        # --------------------------------------------------------------
        # Initial non-negative estimate
        # --------------------------------------------------------------
        try:
            estimate, *_ = nnls(
                rr_matrix,
                observations,
            )
        except RuntimeError:
            estimate = nnls_stable(
                rr_matrix,
                observations,
            )

        estimate = np.clip(
           estimate,
           1e-8,
           None,
        )

        # --------------------------------------------------------------
        # Iteratively reweighted non-negative GLS
        # --------------------------------------------------------------
        for _ in range(max_iter):

            expected_private_count = (
                rr_matrix @ estimate
            )

            observation_covariance = (
                np.diag(expected_private_count)
                / n_days
            )

            condition_number = np.linalg.cond(
                observation_covariance
            )

            regularization_value = max(
                regularization,
                regularization
                * condition_number
                / 1e6,
            )

            regularized_covariance = (
                observation_covariance
                + regularization_value
                * np.eye(n_states)
            )

            try:
                weighting_matrix = np.linalg.inv(
                    regularized_covariance
                )
            except np.linalg.LinAlgError:
                weighting_matrix = np.linalg.pinv(
                    regularized_covariance
                )

            # Cholesky factor used to formulate weighted least squares.
            weighting_sqrt = np.linalg.cholesky(
                weighting_matrix
                + regularization_value
                * np.eye(n_states)
            )

            weighted_matrix = (
                weighting_sqrt.T @ rr_matrix
            )

            weighted_observations = (
                weighting_sqrt.T @ observations
            )

            try:
                new_estimate, *_ = nnls(
                    weighted_matrix,
                    weighted_observations,
                )
            except RuntimeError:
                new_estimate = nnls_stable(
                    weighted_matrix,
                    weighted_observations,
                )

            new_estimate = np.clip(
               new_estimate,
               1e-8,
               None,
            )

            relative_change = (
                np.linalg.norm(new_estimate - estimate)/
                (np.linalg.norm(estimate) + 1e-12)
            )

            estimate = new_estimate
            if relative_change < tolerance:
                break

        # --------------------------------------------------------------
        # Covariance of the final estimator
        # --------------------------------------------------------------
        
        expected_private_count = (rr_matrix @ estimate)
        observation_covariance = (np.diag(expected_private_count)/ n_days)
        condition_number = np.linalg.cond(observation_covariance)

        regularization_value = max(regularization,
            regularization* condition_number/ 1e6,
        )

        regularized_covariance = (
            observation_covariance + regularization_value * np.eye(n_states)
        )

        try:
            weighting_matrix = np.linalg.inv(
                regularized_covariance
            )
        except np.linalg.LinAlgError:
            weighting_matrix = np.linalg.pinv(
                regularized_covariance
            )

        information_matrix = (
            rr_matrix.T
            @ weighting_matrix
            @ rr_matrix
            + regularization_value
            * np.eye(n_states)
        )

        try:
            estimator_covariance = np.linalg.inv(
                information_matrix
            )
        except np.linalg.LinAlgError:
            estimator_covariance = np.linalg.pinv(
                information_matrix
            )

        lambda_hat[:,time_index] = estimate

        covariance[time_index] = 0.5 * (
            estimator_covariance
            + estimator_covariance.T
        )

        if time_index % 5 == 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(
                estimator_covariance,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            plt.colorbar(label="Estimator covariance")
            plt.xlabel("State index")
            plt.ylabel("State index")
            plt.title(f"Estimator covariance (time index {time_index})")
            plt.tight_layout()
            plt.savefig(f"ec_hm_{time_index}_{epsilon}.png", dpi=300)
            plt.close()

        
            

    isflex = 0
    isbulk = 0
    print(f"Total states: {n_states}, Total time steps: {n_time_steps}")
    for time_index in range(n_time_steps):
        for state_index in range(n_states):
            if lambda_hat[state_index, time_index] <= 1e-8:
                #print(f"Zero arrival rate at time step {time_index}, state {states[state_index]}: {lambda_hat[state_index, time_index]}")
                isflex = isflex + 1 if (states[state_index][1] > 0 and states[state_index][0] > 0) else 0
                isbulk = isbulk + 1 if (states[state_index][0] == 0 or states[state_index][1] == 0) else 0
            
    print(f"Flex state count: {isflex}, Bulk state count: {isbulk} ratio {isflex/(isflex+isbulk):.2f}")
    return lambda_hat, covariance


def compute_predictive_arrival_covariance_ldp(
    lambda_hat: np.ndarray,
    Sigma_lambda: np.ndarray,
) -> np.ndarray:
    """
    Combine Poisson variability and estimation uncertainty.

    Parameters
    ----------
    lambda_hat:
        Estimated arrival rates with shape ``(n_states, T)``.
    Sigma_lambda:
        Estimator covariance with shape ``(T, n_states, n_states)``.

    Returns
    -------
    np.ndarray
        Predictive arrival covariance for each time step.
    """
    _, n_time_steps = lambda_hat.shape

    predictive_covariance = np.zeros_like(
        Sigma_lambda,
        dtype=float,
    )

    for time_index in range(n_time_steps):

        poisson_covariance = np.diag(
            lambda_hat[:, time_index]
        )

        predictive_covariance[time_index] = (
            poisson_covariance
            + Sigma_lambda[time_index]
        )

    return predictive_covariance


# ============================================================================
# Arrival-rate estimation from generated records
# ============================================================================

def estimate_arrival_rate_from_records(
    states: list[Tuple[int, int]],
    n_time_steps: int,
    time_step_h: float,
    n_chargers: int = 100,
    privacy_mechanism: Optional[str] = None,
    epsilon: float = 5.0,
    csv_path: Optional[str] = None,
    seed: int = 42,
):
    """
    Estimate state-dependent arrival rates from EV-record data.

    Parameters
    ----------
    states:
        EV state space.
    n_time_steps:
        Number of time steps in the estimation horizon.
    time_step_h:
        Duration of one time step in hours.
    n_chargers:
        Charger-count identifier used for default file selection.
    privacy_mechanism:
        Privacy method: ``"LDP"``, ``"CDP"``, or ``None``.
    epsilon:
        Privacy budget.
    csv_path:
        Input CSV path. If omitted, a default project path is used.
    seed:
        Random seed for stochastic privacy mechanisms.

    Returns
    -------
    tuple
        Estimated arrival rates, states, and privacy-related covariance
        information when applicable.
    """
    print(
        f"Inferring arrival rates: T={n_time_steps},privacy={privacy_mechanism},"
        f"epsilon={epsilon}, dt={time_step_h}"
    )

    # ------------------------------------------------------------------
    # Locate input data
    # ------------------------------------------------------------------
    if csv_path is None:

        if privacy_mechanism == "LDP":
            csv_path = (f"data/ev_records_{n_chargers}_DP_eps{epsilon}_insample.csv")
        else:
            csv_path = (f"data/ev_records_{n_chargers}_insample.csv")

    df = pd.read_csv(csv_path)
    n_days = df["day"].nunique()

    # ------------------------------------------------------------------
    # Empirical arrival counts
    # ------------------------------------------------------------------
    observed_arrival_rate = np.zeros(
        (len(states), n_time_steps),
        dtype=float,
    )

    for (state_index, time_step), group in df.groupby(["state_idx", "time_step"]):

        if (state_index >= len(states) or time_step >= n_time_steps):
            continue

        observed_arrival_rate[int(state_index),int(time_step),] = (
            group.groupby("day")
            .size()
            .reindex(
                df["day"].unique(),
                fill_value=0,
            )
            .mean()
        )

    print(f"Arrival-rate matrix shape: {observed_arrival_rate.shape}")

    # ------------------------------------------------------------------
    # LDP
    # ------------------------------------------------------------------
    if privacy_mechanism == "LDP":

        rr_matrix = build_rr_matrix(
            epsilon=epsilon,
            n_states=len(states),
        )

        lambda_hat, Sigma_lambda = (
            estimate_arrival_rate_ldp(
                rr_matrix=rr_matrix,
                privatized_arrival_rate=observed_arrival_rate,
                states=states,
                epsilon=epsilon,
                n_days=n_days,
            )
        )

        Sigma_tilde_a = (
            compute_predictive_arrival_covariance_ldp(
                lambda_hat=lambda_hat,
                Sigma_lambda=Sigma_lambda,
            )
        )

        return (lambda_hat, states, Sigma_tilde_a, Sigma_lambda,)

    # ------------------------------------------------------------------
    # CDP
    # ------------------------------------------------------------------
    if privacy_mechanism == "CDP":

        return estimate_arrival_rate_cdp(
            observed_arrival_rate=observed_arrival_rate,
            df=df,
            states=states,
            n_time_steps=n_time_steps,
            n_days=n_days,
            epsilon=epsilon,
            seed=seed,
        )

    return observed_arrival_rate, states


# ============================================================================
# CDP estimator
# ============================================================================

def estimate_arrival_rate_cdp(
    observed_arrival_rate: np.ndarray,
    df: pd.DataFrame,
    states: list[Tuple[int, int]],
    n_time_steps: int,
    n_days: int,
    epsilon: float,
    seed: int = 42,
):
    """
    Estimate arrival rates using a central-DP discrete-Laplace mechanism.

    Independent discrete-Laplace noise is added to daily state-arrival
    counts before averaging and projecting onto the non-negative orthant.

    Parameters
    ----------
    observed_arrival_rate:
        Empirical state-dependent arrival rates.
    df:
        EV-record dataframe containing day, state, and time-step columns.
    states:
        EV state space.
    n_time_steps:
        Number of time steps in the horizon.
    n_days:
        Number of observation days.
    epsilon:
        Central-DP privacy budget.
    seed:
        Random seed for noise generation.

    Returns
    -------
    tuple
        Private arrival-rate estimates, states, predictive covariance,
        and estimator covariance.
    """
    rng = np.random.default_rng(seed)

    n_states = len(states)

    # ------------------------------------------------------------------
    # Recover true daily arrival counts
    # ------------------------------------------------------------------
    days = sorted(df["day"].unique())

    daily_arrivals = np.zeros(
        (n_days, n_states, n_time_steps),
        dtype=float,
    )

    for (state_index, time_step, day,), group in df.groupby(["state_idx", "time_step", "day"]):

        if (state_index >= n_states or time_step >= n_time_steps):
            continue

        day_index = days.index(day)

        daily_arrivals[day_index,int(state_index),int(time_step),] = len(group)

    # ------------------------------------------------------------------
    # Discrete-Laplace mechanism
    # ------------------------------------------------------------------
    sensitivity = 2.0
    q = np.exp(-epsilon / sensitivity)

    noise_variance = (
        2.0 * q / (1.0 - q) ** 2
    )

    def sample_discrete_laplace(
        rng: np.random.Generator,
        q: float,
        size: tuple,
    ) -> np.ndarray:
        """
        Sample a discrete-Laplace random variable.
        """
        g1 = (rng.geometric(1.0 - q,size=size,)- 1)
        g2 = (rng.geometric(1.0 - q,size=size,)- 1)

        return g1 - g2

    # ------------------------------------------------------------------
    # Add independent DP noise to each day
    # ------------------------------------------------------------------
    noisy_arrivals = []

    for day_index in range(n_days):

        noise = sample_discrete_laplace(rng=rng,q=q,size=(n_states, n_time_steps),)
        noisy_arrivals.append(daily_arrivals[day_index] + noise)

    noisy_arrivals = np.asarray(noisy_arrivals)

    # ------------------------------------------------------------------
    # Average over historical days
    # ------------------------------------------------------------------
    raw_lambda = np.mean(noisy_arrivals,axis=0)

    # ------------------------------------------------------------------
    # Non-negative plug-in estimator
    # ------------------------------------------------------------------
    lambda_hat = np.maximum(raw_lambda,0.0)

    # ------------------------------------------------------------------
    # Estimator covariance
    # ------------------------------------------------------------------
    Sigma_lambda = [
        (
            np.diag(lambda_hat[:, time_index])
            + noise_variance
            * np.eye(n_states)
        )
        / n_days
        for time_index in range(n_time_steps)
    ]

    # ------------------------------------------------------------------
    # Future Poisson covariance
    # ------------------------------------------------------------------
    Sigma_poisson = [
        np.diag(lambda_hat[:, time_index])
        for time_index in range(n_time_steps)
    ]

    # ------------------------------------------------------------------
    # Predictive covariance
    # ------------------------------------------------------------------
    Sigma_tilde_a = [
        Sigma_poisson[time_index]+ Sigma_lambda[time_index]
        for time_index in range(n_time_steps)
    ]

    return (
        lambda_hat,
        states,
        Sigma_tilde_a,
        Sigma_poisson,
    )


# ============================================================================
# Randomized response
# ============================================================================

def build_rr_matrix(
    epsilon: float,
    n_states: int,
) -> np.ndarray:
    """
    Construct the K-ary randomized-response transition matrix.

    Parameters
    ----------
    epsilon : float
        LDP privacy parameter.
    n_states : int
        Number of states K.

    Returns
    -------
    np.ndarray
        K x K randomized-response matrix.
    """
    if epsilon < 0:
        raise ValueError( "epsilon must be non-negative.")

    if n_states <= 0:
        raise ValueError("n_states must be positive.")

    denominator = (np.exp(epsilon) + n_states - 1)

    probability_self = (np.exp(epsilon)/ denominator)

    probability_other = (1.0 / denominator)

    rr_matrix = np.full((n_states, n_states),probability_other,)

    np.fill_diagonal(rr_matrix,probability_self)

    return rr_matrix

