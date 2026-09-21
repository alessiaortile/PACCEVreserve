import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm, poisson
import pickle
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from lam_price_setting import estimate_arrival_rate_from_records, load_ev_model_parameters, read_daily_prices
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ParsedPolicyOutput:
    """
    Container storing the optimization outputs.
    Attributes
    ----------
    bid_up : np.ndarray - Hourly upward reserve bids.
    bid_down : np.ndarray - Hourly downward reserve bids.
    objective_value : np.ndarray - Objective value of the optimization problem.
    reliability : np.ndarray - Reliability metrics computed during evaluation.
    lam : np.ndarray - Estimated state-dependent arrival rates.
    status : list[tuple[int, int]] - EV state space representation (remaining charge, slack).
    hour_labels : list[int] - Hour indices.
    time_labels : list[int] - Optimization time-step indices.
    z : np.ndarray - Charging flow variables.
    y : np.ndarray - Waiting flow variables.
    Mz : dict, optional - Charging disturbance-feedback coefficients.
    My : dict, optional - Waiting disturbance-feedback coefficients.
    """
    bid_up: np.ndarray
    bid_down: np.ndarray
    objective_value: np.ndarray
    reliability: np.ndarray
    lam: np.ndarray
    status: List[Tuple[int, int]]
    hour_labels: List[int]
    time_labels: List[int]
    z: np.ndarray
    y: np.ndarray
    Mz: Dict[Tuple[int,int,int,int], float] | None = None
    My: Dict[Tuple[int,int,int,int], float] | None = None


def build_state_predecessor_maps(status):
    """
    Build lookup tables for EV state transitions.

    For each state `(charge, slack)`, this function identifies the
    predecessor states that can reach it after one time step through:

    - Charging:
        (charge + 1, slack) -> (charge, slack)

    - Waiting:
        (charge, slack + 1) -> (charge, slack)

    Parameters
    ----------
    status : list[tuple[int, int]]
        List of EV states, where each state is represented by
        `(remaining_charge, slack_time)`.

    Returns
    -------
    charge_predecessors : defaultdict[list]
        Mapping from a destination state `(charge, slack)` to the
        indices of states that reach it after one charging action.

    wait_predecessors : defaultdict[list]
        Mapping from a destination state `(charge, slack)` to the
        indices of states that reach it after one waiting action.
    """
    charge_predecessors = defaultdict(list)
    wait_predecessors = defaultdict(list)
    for j, (charge, slack) in enumerate(status):
        charge_predecessors[(charge-1, slack)].append(j)
        wait_predecessors[(charge, slack-1)].append(j)
    return charge_predecessors, wait_predecessors


def reachable_states(
    arrival_state: tuple[int, int],
    tau: int,
) -> list[tuple[int, int]]:
    """
    Compute all EV states reachable after `tau` time steps.
    Parameters
    ----------
    arrival_state :
        Initial EV state (remaining_charge, slack_time).
    tau :
        Number of elapsed time steps.
    Returns
    -------
    list[tuple[int, int]]
        Reachable states after tau transitions.
    """
    
    c0, s0 = arrival_state
    states = []

    imin = max(0, tau - s0)
    imax = min(tau, c0)

    for i in range(imin, imax + 1):
        c = c0 - i
        s = s0 - (tau - i)
        states.append((c, s))

    return states


def build_active_states(status, lam, T):
    """
    Identify reachable state transitions with nonzero arrivals.

    Parameters
    ----------
    status : list[tuple[int, int]]
        EV state space (remaining charge, slack time).

    lam : np.ndarray
        Expected arrival rates for each state and time step.

    T : int
        Optimization horizon.

    Returns
    -------
    Mactive : list[tuple[int, int, int, int]]
        Active disturbance-feedback indices (t, k, j, i).

    Reach : dict
        Maps (t, k, j) to the set of reachable state indices.
    """
    state_to_index = {s: i for i, s in enumerate(status)}

    Mactive = []
    Reach = {}

    for k in range(T):
        active_arrivals = np.where(lam[:, k] > 0)[0]

        for j in active_arrivals:
            x0 = status[j]

            for t in range(k, T):
                tau = t - k
                reach = []

                for x in reachable_states(x0, tau):
                    if x not in state_to_index:
                        continue

                    c, s = x
                    if c == 0 or s == 0:
                        continue

                    i_dest = state_to_index[x]
                    reach.append(i_dest)
                    Mactive.append((t, k, j, i_dest))

                Reach[(t, k, j)] = reach

    return Mactive, Reach


def deviation_bounds(lam, alpha):
    """
    Compute lower and upper Poisson arrival-count deviations.
    Parameters
    ----------
    lam:
        Expected arrival rates. 
    alpha:
        Central coverage probability of the interval.
    Returns
    -------
    lower_deviation:
        Difference between the lower Poisson quantile and ``lam``.
    upper_deviation:
        Difference between the upper Poisson quantile and ``lam``.
    """
    lam = np.asarray(lam, dtype=float)
    safe_lam = lam.copy()

    lower_tail = (1.0 - alpha) / 2.0
    upper_tail = 1.0 - (1.0 - alpha) / 2.0

    a_lo = poisson.ppf(lower_tail, safe_lam)
    a_hi = poisson.ppf(upper_tail, safe_lam)

    x = a_lo - safe_lam
    v = a_hi - safe_lam

    return x, v
# ----------------------------------------------------------------------
# Main co-optimization
# ----------------------------------------------------------------------

def arrival_bias_propagation(status,T,lam,z,y,eps,next_charge,next_slack,
    flex_state,p_rate,r_up_x,r_down_x,kappa_h,tau_up,tau_down,T_h,H,price_up,price_down,):
    """
    Estimate arrival-induced reserve bias and optimize bias-adjusted bids.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Bias-adjusted upward and downward reserve bids.
    """

    # compute charging policy u[i,t] = z[i,t]/(z[i,t]+y[i,t]) for each state i and time t
    n_states = len(status)
    u = np.zeros((n_states, T), dtype=float)

    for i in range(n_states):
        denom = y[i,:] + z[i,:]
        ratio = np.zeros_like(denom, dtype=float)
        valid = denom > 0
        ratio[valid] = z[i, valid] / denom[valid]
        ratio[ratio <= 0] = 0
        u[i, :] = ratio

        if status[i][0] > 0 and status[i][1] == 0:
            u[i,:] = 1.0
        elif status[i][0] == 0:
            u[i,:] = 0.0
    
    u = np.clip(u, 0.0, 1.0)

    # compute bias propagation through the population dynamics

    z_bar = np.zeros((n_states, T))
    y_bar = np.zeros((n_states, T))

    n_days = 60
    Delta1 = 2.0
    q = np.exp(-eps / Delta1)
    sigma_eta2 = 2.0 * q / (1.0 - q)**2

    s_hat = np.sqrt((lam + sigma_eta2) / n_days)
    b_hat = {(j, t): s_hat[j][t] / np.sqrt(2 * np.pi) for j in range(len(status)) for t in range(T)}

    n_current = [b_hat[j, 0] for j in range(len(status))]
    for t in range(T):
        u_t = u[:, t]
        z_bar[:, t] = u_t * n_current
        y_bar[:, t] = (1.0 - u_t) * n_current

        if t < T - 1:
            B = np.zeros((n_states, n_states), dtype=float)
            for i, (charge, slack) in enumerate(status):
                for j in next_charge.get((charge, slack), []):
                    B[i, j] = u_t[j]
                for j in next_slack.get((charge, slack), []):
                    B[i, j] = 1.0 - u_t[j]
            n_current = B @ n_current + [b_hat[j, t + 1] for j in range(len(status))]

    r_up_bias = np.zeros(T)
    r_down_bias = np.zeros(T)
    for t in range(T):
        r_up_bias[t] = sum(z_bar[i, t] for i in flex_state)*p_rate
        r_down_bias[t] = sum(y_bar[i, t] for i in flex_state)*p_rate

    # Recompute optimal bid using the bias-adjusted reserve capacities
    m = gp.Model()
    bidu = m.addVars(H, lb=0)
    bidd = m.addVars(H, lb=0)

    tot_reserve_up = np.array([r_up_x[t] - r_up_bias[t] - p_rate * kappa_h[int(t*dt)] * tau_up[int(t*dt), t, "up"] if r_up_x[t] - r_up_bias[t] - p_rate * kappa_h[int(t*dt)] * tau_up[int(t*dt), t, "up"]  > 0 else 0 for t in range(T)]) 
    tot_reserve_down = np.array([r_down_x[t] - r_down_bias[t]  - p_rate * kappa_h[int(t*dt)] * tau_down[int(t*dt), t, "down"] if r_down_x[t] - r_down_bias[t]  - p_rate * kappa_h[int(t*dt)] * tau_down[int(t*dt), t, "down"] > 0 else 0 for t in range(T)])

    m.addConstrs(bidu[h] + bidd[h]*0.2 <= tot_reserve_up[t]  for h in range(H) for t in T_h[h])
    m.addConstrs(bidd[h] <= tot_reserve_down[t]  for h in range(H) for t in T_h[h])

    objective = gp.quicksum(
        price_up[h] * bidu[h] + price_down[h] * bidd[h] for h in range(H)
    )

    m.setObjective(objective, GRB.MAXIMIZE)
    m.optimize()

    new_bid_up = np.array([bidu[h].X for h in range(H)])
    new_bid_down = np.array([bidd[h].X for h in range(H)])
    bid_up = new_bid_up
    bid_down = new_bid_down

    return bid_up, bid_down

def cooptimization(lam,sigma, status, flex_state, price_up, price_down, real_price,
                    dt, n, T, H, p_rate, lic_path,eps,DP, beta=0.1):
    """
    Solve the robust EV charging and reserve co-optimization problem.

    Parameters
    ----------
    lam : array-like
        Arrival rates for each state.
    sigma : array-like
        Uncertainty estimates for each state.
    status : list of tuples
        List of (charge, slack) pairs representing the system states.
    flex_state : list of int
        Indices of flexible states.
    price_up : array-like
        Upward reserve prices.
    price_down : array-like
        Downward reserve prices.
    real_price : array-like
        Real-time prices.
    dt : float
        Time step size.
    n : int
        Number of EVs.
    T : int
        Number of time steps.
    H : int
        Number of market hours.
    p_rate : float
        Power rate.
    lic_path : str
        Path to the license file.
    eps : float
        Epsilon value for the robust optimization.
    DP : bool
        Whether to use dynamic programming.
    beta : float, optional (default=0.1)
        Confidence level for the Bonferroni correction.

    Returns
    -------
    tuple
        Charging flows, waiting flows, reserve bids, objective value,
        reserve capacities, and disturbance-feedback coefficients.
    """

    arrival_deviation_lower, arrival_deviation_upper = deviation_bounds(lam, alpha=0.95)
    next_charge, next_slack = build_state_predecessor_maps(status)
    idx = {(c, s): i for i, (c, s) in enumerate(status)}

    # ---- sparse support of the disturbance-feedback matrices ----
    Mactive, Reach = build_active_states(status, lam, T)
    Mactive_set = set(Mactive)

    # ---- predecessor maps used by the achievability recursion (Eq. 46) ----
    wait_pred = [None] * len(status)
    charge_pred = [None] * len(status)
    for i, (charge, slack) in enumerate(status):
        wait_pred[i] = idx.get((charge, slack + 1))
        charge_pred[i] = idx.get((charge + 1, slack))

    flex_state = set(flex_state)  # convert to set for faster membership testing

    mandatory_charge_states = set(i for i, (c, s) in enumerate(status) if s == 0 and c > 0)
    full_charge_states = set(i for i, (c, s) in enumerate(status) if c == 0 and s > 0)

    # ---- market-hour <-> intra-hour time-step mapping ----
    T_h = [[] for _ in range(H)]
    for t in range(T):
        T_h[int(t * dt)].append(t)

    # ---- list of (k, j) arrival pairs that actually carry mass ----
    active_arrivals = [(k, j) for k in range(T) for j in range(len(status)) if lam[j, k] > 0]

    # ---- bonferroni radius per hour ----
    kappa_h = {h: norm.ppf(1 - beta/2) for h in range(H)}

    m = gp.Model()
    m.setParam('OutputFlag', 1)
    m.setParam("Threads", min(8, os.cpu_count() or 1))
    m.setParam("Method", 2)    

    # variable definition
    z_bar = m.addVars(len(status), T, lb=0, name="z_bar")
    y_bar = m.addVars(len(status), T, lb=0, name="y_bar")

    Mz = m.addVars(Mactive, lb=0, name="Mz")
    My = m.addVars(Mactive, lb=0, name="My")

    r_up = m.addVars(T, lb=0, name="r_up")
    r_down = m.addVars(T, lb=0, name="r_down")

    b_up = m.addVars(H, lb=0, name="b_up")
    b_down = m.addVars(H, lb=0, name="b_down")

    u_z = m.addVars(Mactive, lb=-GRB.INFINITY, name="u_z")
    u_y = m.addVars(Mactive, lb=-GRB.INFINITY, name="u_y")

    tau = {}
    for h in range(H):
        for t in T_h[h]:
            tau[h, t, "up"] = m.addVar(
                lb=0, name=f"tau_h{h}_t{t}_up"
            )
            tau[h, t, "down"] = m.addVar(
                lb=0, name=f"tau_h{h}_t{t}_down"
            )

    # Population dynamics, plus mandatory-action limits

    for t in range(T):
        for i, (charge, slack) in enumerate(status):
            if t == 0:
                incoming = lam[i, t]
            else:
                incoming = (
                    lam[i, t]
                    + gp.quicksum(z_bar[j, t - 1] for j in next_charge[(charge, slack)])
                    + gp.quicksum(y_bar[j, t - 1] for j in next_slack[(charge, slack)])
                )

            m.addConstr(
                z_bar[i, t] + y_bar[i, t] == incoming,
                name=f"population_balance_t{t}_state{i}",
            )

            if i in mandatory_charge_states:
                m.addConstr(
                    y_bar[i, t] == 0,
                    name=f"mandatory_charge_t{t}_state{i}",
                )

            if i in full_charge_states:
                m.addConstr(
                    z_bar[i, t] == 0,
                    name=f"fully_charged_t{t}_state{i}",
                )

    # Add constraints for the flow robustness 

    m.addConstrs(
        (
            u_z[t, k, j, i] <= Mz[t, k, j, i] * arrival_deviation_lower[j, k]
            for (t, k, j, i) in Mactive
        ),
        name="uz_lower_deviation_bound",
    )

    m.addConstrs(
        (
            u_z[t, k, j, i] <= Mz[t, k, j, i] * arrival_deviation_upper[j, k]
            for (t, k, j, i) in Mactive
        ),
        name="uz_upper_deviation_bound",
    )

    m.addConstrs(
        (
            u_y[t, k, j, i] <= My[t, k, j, i] * arrival_deviation_lower[j, k]
            for (t, k, j, i) in Mactive
        ),
        name="uy_lower_deviation_bound",
    )

    m.addConstrs(
        (
            u_y[t, k, j, i] <= My[t, k, j, i] * arrival_deviation_upper[j, k]
            for (t, k, j, i) in Mactive
        ),
        name="uy_upper_deviation_bound",
    )

    # ---- group Mactive entries by (t, i) ----
    terms_by_ti = defaultdict(list)
    for (t, k, j, i) in Mactive:
        terms_by_ti[(t, i)].append((k, j))

    for t in range(T):
        for i in range(len(status)):
            terms = terms_by_ti.get((t, i))
            if terms:  # skip entirely if this (i,t) has no disturbance exposure
                m.addConstr(
                    z_bar[i, t]
                    + gp.quicksum(u_z[t, k, j, i] for (k, j) in terms)
                    >= 0,
                    name=f"robust_z_nonnegative_t{t}_state{i}",
                )

                m.addConstr(
                    y_bar[i, t]
                    + gp.quicksum(u_y[t, k, j, i] for (k, j) in terms)
                    >= 0,
                    name=f"robust_y_nonnegative_t{t}_state{i}",
                )

    for (t, k, j, i) in Mactive:

        if t == k:

            rhs = 1.0 if i == j else 0.0

        else:

            rhs = gp.LinExpr()

            p = charge_pred[i]

            if p is not None and (t - 1, k, j, p) in Mz:

                rhs += Mz[t - 1, k, j, p]

            p = wait_pred[i]

            if p is not None and (t - 1, k, j, p) in My:

                rhs += My[t - 1, k, j, p]

        m.addConstr(
            Mz[t, k, j, i] + My[t, k, j, i] == rhs,
            name=f"rec_{t}_{k}_{j}_{i}",
        )

    m.addConstrs(
        (
            My[t, k, j, i] == 0
            for (t, k, j, i) in Mactive
            if i in mandatory_charge_states
        ),
        name="feedback_mandatory_charge",
    )

    m.addConstrs(
        (
            Mz[t, k, j, i] == 0
            for (t, k, j, i) in Mactive
            if i in full_charge_states
        ),
        name="feedback_fully_charged",
    )

    # Reserve definitions
    print("Building reserve definitions...")
    for t in range(T):
        m.addConstr(
            r_up[t] == gp.quicksum(z_bar[i, t] for i in flex_state) * p_rate,
            name=f"upward_reserve_definition_t{t}",
        )
        m.addConstr(
            r_down[t] == gp.quicksum(y_bar[i, t] for i in flex_state) * p_rate,
            name=f"downward_reserve_definition_t{t}",
        )


    flex_state_set = set(flex_state)

    # Precompute active_by_k 
    active_by_k = {}
    for (k, j) in active_arrivals:
        active_by_k.setdefault(k, []).append(j)


    reach_flex = {}
    for (k, j) in active_arrivals:
        for t in range(T):
            ri = Reach.get((t, k, j))
            if not ri:
                continue
            filtered = [i for i in ri if i in flex_state_set]
            if filtered:
                reach_flex[t, k, j] = filtered

    # For each t, which k's actually have *any* nonzero contribution?
    relevant_k_for_t = {t: set() for t in range(T)}
    for (t, k, j) in reach_flex:
        relevant_k_for_t[t].add(k)

    # Precompute sparse nonzero (j1,j2) pairs of sigma_k ONCE
    nonzero_pairs_by_k = {}
    for k, js in active_by_k.items():
        sigma_k = sigma[k]
        pairs = [(j1, j2) for j1 in js for j2 in js ] 
        if pairs:
            nonzero_pairs_by_k[k] = pairs

    # ---- sens_up / sens_down, built only where nonzero --------------------

    sens_up = {}
    sens_down = {}
    for (t, k, j), ilist in reach_flex.items():
        sens_up[t, k, j] = gp.quicksum(Mz[t, k, j, i] for i in ilist)
        sens_down[t, k, j] = gp.quicksum(My[t, k, j, i] for i in ilist)

    # ---- main loop: only relevant k, only nonzero sigma pairs --------------

    qexpr_up = [gp.QuadExpr() for t in range(T)]
    qexpr_down = [gp.QuadExpr() for t in range(T)]

    for t in range(T):
        for k in relevant_k_for_t[t]:
            pairs = nonzero_pairs_by_k.get(k)
            if not pairs:
                continue
            sigma_k = sigma[k]
            for j1, j2 in pairs:
                s1_up = sens_up.get((t, k, j1))
                s2_up = sens_up.get((t, k, j2))
                if s1_up is not None and s2_up is not None:
                    qexpr_up[t] += sigma_k[j1][j2] * s1_up * s2_up

                s1_dn = sens_down.get((t, k, j1))
                s2_dn = sens_down.get((t, k, j2))
                if s1_dn is not None and s2_dn is not None:
                    qexpr_down[t] += sigma_k[j1][j2] * s1_dn * s2_dn

        h = int(t * dt)
        m.addQConstr(tau[h, t, "up"] * tau[h, t, "up"] >= qexpr_up[t],
                    name=f"upward_soc_norm_h{h}_t{t}")
        m.addQConstr(tau[h, t, "down"] * tau[h, t, "down"] >= qexpr_down[t],
                    name=f"downward_soc_norm_h{h}_t{t}")


    # SOC feasibility (ellipsoid contained in reserve-feasible polytope),
    for h in range(H):
        kappa = kappa_h[h]
        for t in T_h[h]:
            m.addConstr(
                r_up[t] - p_rate * kappa * tau[h, t, "up"] 
                >= b_up[h] + 0.2 * b_down[h],
                name=f"upward_reserve_feasibility_h{h}_t{t}",
            )
            m.addConstr(
                r_down[t] - p_rate * kappa * tau[h, t, "down"] 
                >= b_down[h],
                name=f"downward_reserve_feasibility_h{h}_t{t}",
            )


    # Objective, 
    objective = gp.quicksum(
        price_up[h] * b_up[h] + price_down[h] * b_down[h] for h in range(H)
    ) - gp.quicksum(
        real_price[t] * z_bar[i, t] * p_rate*dt for t in range(T) for i in range(len(status))
    )
 
    m.setObjective(objective, GRB.MAXIMIZE)

    m.setParam("DualReductions", 0)
    print("NumBinVars:", m.NumBinVars)
    print("NumIntVars:", m.NumIntVars)
    print("NumSOS:", m.NumSOS)
    print("NumQConstrs:", m.NumQConstrs)
    print("NumConstrs:", m.NumConstrs)
    m.optimize()
    print("objective value:", m.objVal)
    print("Reserve bid status:", m.status)
   
    z = np.array([[z_bar[i, t].X for t in range(T)] for i in range(len(status))])
    y = np.array([[y_bar[i, t].X for t in range(T)] for i in range(len(status))])
    bid_up = np.array([b_up[h].X for h in range(H)])
    bid_down = np.array([b_down[h].X for h in range(H)])
    bid_rev = m.objVal
    r_up_x = np.array([r_up[t].X for t in range(T)])
    r_down_x = np.array([r_down[t].X for t in range(T)])
    Mz_sol = {(t, k, j, i): Mz[t, k, j, i].X for (t, k, j, i) in Mactive}
    My_sol = {(t, k, j, i): My[t, k, j, i].X for (t, k, j, i) in Mactive}
    tau_up= {(int(t*dt), t, "up"): tau[int(t*dt), t, "up"].X for t in range(T)}
    tau_down = {(int(t*dt), t, "down"): tau[int(t*dt), t, "down"].X for t in range(T)}

    print("tau_up:", tau_up)
    print("tau_down:", tau_down)
    print("qexpress_up:", [qexpr_up[t].getValue() for t in range(T)])
    print("qexpress_down:", [qexpr_down[t].getValue() for t in range(T)])
    
    # to comment to avoid PP 
    if DP == "CDP":
        bid_up, bid_down = arrival_bias_propagation(status,T,lam,z,y,eps,next_charge,next_slack,
            flex_state,p_rate,r_up_x,r_down_x,kappa_h,tau_up,tau_down,T_h,H,price_up,price_down)

    return z, y, bid_up, bid_down, bid_rev, r_up_x, r_down_x, Mz_sol, My_sol

def error_log(lam, real_lam, sigma, status, T):
    """Print estimation-error and z-score diagnostics for the first ``T`` steps.
    Parameters
    ----------
    lam : array-like
        Estimated arrival rates for each state.
    real_lam : array-like
        True arrival rates for each state.
    sigma : array-like
        Covariance matrices for the estimated arrival rates.
    status : list of tuples
        List of (charge, slack) pairs representing the system states.
    T : int
        Number of time steps to consider for diagnostics."""
    
    lam = np.asarray(lam)[:, :T]
    real_lam = np.asarray(real_lam)[:, :T]

    flex_state = [
        i for i, (charge, slack) in enumerate(status)
        if charge > 0 and slack > 0
    ]

    std = np.array([
        [np.sqrt(max(float(sigma[t][j][j]), 0.0)) for t in range(T)]
        for j in flex_state
    ])

    z = np.divide(
        lam[flex_state] - real_lam[flex_state],
        std,
        out=np.zeros_like(lam[flex_state], dtype=float),
        where=std > 0,
    )

    print("mean z-score across all states and time steps:", np.mean(np.abs(z)))
    print("max z-score across all states and time steps:", np.max(np.abs(z)))

    positive_z = np.maximum(z, 0)
    z_max = np.max(np.abs(positive_z), axis=1)

    print(
        "mean positive z-score across all states and time steps:",
        np.mean(positive_z),
    )

    print("States with the largest z-scores:")
    for idx in np.argsort(z_max)[::-1][:10]:
        state_idx = flex_state[idx]
        charge, slack = status[state_idx]
        print(
            f"State {state_idx}: max z-score = {z_max[idx]:.2f}, "
            f"charge = {charge}, slack = {slack}"
        )

    energy_hat = np.sum(lam * np.array([charge for charge, _ in status])[:, None])
    energy_real = np.sum(real_lam * np.array([charge for charge, _ in status])[:, None])

    slack_hat = np.sum(lam * np.array([slack for _, slack in status])[:, None])
    slack_real = np.sum(real_lam * np.array([slack for _, slack in status])[:, None])

    total_hat = np.sum(lam)
    total_real = np.sum(real_lam)

    flex_hat = np.sum(lam[flex_state])
    flex_real = np.sum(real_lam[flex_state])

    def error_percent(value, reference):
        return 100 * (value - reference) / reference if reference else 0.0

    print(
        f"Total energy (hat): {energy_hat}, "
        f"Total energy (real): {energy_real} "
        f"%error: {error_percent(energy_hat, energy_real):.2f}%"
    )
    print(
        f"Total slack (hat): {slack_hat}, "
        f"Total slack (real): {slack_real} "
        f"%error: {error_percent(slack_hat, slack_real):.2f}%"
    )
    print(
        f"Total mass (hat): {total_hat}, "
        f"Total mass (real): {total_real} "
        f"%error: {error_percent(total_hat, total_real):.2f}%"
    )
    print(
        f"Total mass (hat) for flex states: {flex_hat}, "
        f"Total mass (real) for flex states: {flex_real} "
        f"%error: {error_percent(flex_hat, flex_real):.2f}%"
    )

    for t in range(T):
        print(f"Flex mass at t={t}: {flex_hat if T == 1 else np.sum(lam[flex_state, t])}, "
              f"{np.sum(real_lam[flex_state, t])}")

if __name__ == "__main__":

    lic_path = "psw/grb.lic"  # Path to your license file

    # Setup
    dt = 1
    H = 24
    T = int(H / dt)
    p_rate = 12
    seed = 42
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description="Optimal bid computation")
    parser.add_argument("--eps", type=float, default=None, help="Epsilon value for lambda inference.")
    parser.add_argument("--scale", type=int, default=100, help="Scaling factor for lambda inference.")
    parser.add_argument("--path", type=Path, default=None, help="Path to load control policy from.")
    parser.add_argument("--DP", type=str, default=None, help="Whether to apply DP state privatization (build_DPevent_records)")
    args = parser.parse_args()
    eps = args.eps
    scale_stats = args.scale
    DP = args.DP

    daily_prices = read_daily_prices("data/fcrd_price.csv","data/spot_price.csv")

    # Stack hourly profiles
    spot_matrix = np.stack([daily_prices[d]["spot"] for d in range(60)],axis=0)

    fcrd_down_matrix = np.stack([daily_prices[d]["fcrd_down"] for d in range(60)],axis=0)

    fcrd_up_matrix = np.stack([daily_prices[d]["fcrd_up"] for d in range(60)],axis=0)

    # Mean hourly prices across the 60 sampled days
    price_el = np.repeat(spot_matrix.mean(axis=0), int(1/dt))
    price_bup = fcrd_down_matrix.mean(axis=0)
    price_bdown = fcrd_up_matrix.mean(axis=0)


    if DP == "LDP":
        print("Loading lambda inference results...")
        try:
            filename = f"data/lambda_inference_{dt:.2f}_{eps}_{H}_{int(scale_stats)}.pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)

            # Convert back to numpy arrays
            time_bin = np.array(data["results"]["time_bin"])
            lam_hat = np.array(data["results"]["lambda_hat"])
            lam = lam_hat[:, :T]  # Use only the first T steps for policy optimization
            real_lam = np.array(data["results"]["lambda_real"])
            sigma = data["metadata"]["sigma"]
            status = [tuple(s) for s in data["metadata"]["status"]]

            error_log(lam, real_lam, sigma, status, T)

        except FileNotFoundError:
            print("Read records and infer lambda...")
            _, status = load_ev_model_parameters(dt, p_rate, scale_stats)

            print("Inferring lambda with DP...")
            lam_hat, status, sigma, Sigma_lambda = (
                estimate_arrival_rate_from_records(
                    states=status,
                    n_time_steps=T,
                    time_step_h=dt,
                    n_chargers=scale_stats,
                    privacy_mechanism="LDP",
                    epsilon=eps,
                )
            )
            print("Lambda inference completed.")
            lam = lam_hat[:, :T]
            real_lam, _ = estimate_arrival_rate_from_records(
                states=status,
                n_time_steps=T,
                time_step_h=dt,
                n_chargers=scale_stats,
                privacy_mechanism=None,
            )

            data = {
                "results": {
                    "time_bin": np.arange(T).tolist(),
                    "lambda_hat": lam_hat[:, :T].tolist(),
                    "lambda_real": real_lam[:, :T].tolist(),
                },
                "metadata": {
                    "sigma": np.array(sigma).tolist(),
                    "status": np.array(status).tolist(),
                }
            }

            filename = f"data/lambda_inference_{dt:.2f}_{eps}_{H}_{int(scale_stats)}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved results to {filename}")

    elif DP == "CDP":
        _, status = load_ev_model_parameters(dt, p_rate, scale_stats)
        lam, status, sigma, Sigma_lambda = (
            estimate_arrival_rate_from_records(
                states=status,
                n_time_steps=T,
                time_step_h=dt,
                n_chargers=scale_stats,
                privacy_mechanism="CDP",
                epsilon=eps,
                csv_path=f"data/ev_records_{int(scale_stats)}_insample.csv",
            )
        )
        real_lam, _ = estimate_arrival_rate_from_records(
            states=status,
            n_time_steps=T,
            time_step_h=dt,
            n_chargers=scale_stats,
            privacy_mechanism=None,
            csv_path=f"data/ev_records_{int(scale_stats)}_insample.csv",
        )
        # for NoPP method
        #sigma = np.sigma = [np.diag(lam[:, t]) for t in range(T)]  # Placeholder for sigma if not using DP propapgatio

        n_days = 60 # training days used for lambda inference
        Delta1 = 2.0
        q = np.exp(-eps / Delta1)
        sigma_eta2 = 2.0 * q / (1.0 - q)**2
        s_hat = np.sqrt((lam + sigma_eta2) / n_days)
        b_hat = {(j, t): s_hat[j][t] / np.sqrt(2 * np.pi) for j in range(len(status)) for t in range(T)}
        lam_ubias = lam - np.array([[b_hat[j, t] for t in range(T)] for j in range(len(status))])
        lam_ubias[lam_ubias < 0] = 0
        
        error_log(lam_ubias, real_lam, sigma, status, T)

    else:
        _, status = load_ev_model_parameters(dt, p_rate, scale_stats)
        lam, status = estimate_arrival_rate_from_records(
            states=status,
            n_time_steps=T,
            time_step_h=dt,
            n_chargers=scale_stats,
            privacy_mechanism=None,
            csv_path=f"data/ev_records_{int(scale_stats)}_insample.csv",
        )
        lam = lam[:,:T]
        sigma = [np.diag(lam[:, t]) for t in range(T)]  # Placeholder for sigma if not using DP #+1.96*se[:, t]
        eps = None  # No DP, so set eps to None

    n = len(status)
    next_charge, next_slack = build_state_predecessor_maps(status)
    flex_state = [i for i, (charge, slack) in enumerate(status) if charge > 0 and slack > 0]
    beta = 0.1  

    z,y,bid_up, bid_down, bid_rev, r_up_x, r_down_x, Mz_sol, My_sol = cooptimization(
        lam, sigma,status, flex_state, price_bup, price_bdown, price_el, dt, n, T, H, p_rate, lic_path,eps, DP=DP, beta=beta
    )

    print("Optimal bids computed:")
    print("Bid up:", bid_up)
    print("Bid down:", bid_down)
    print("Expected revenue:", bid_rev)

    bids = []
    for h in range(H):
        bids.append((bid_up[h], bid_down[h]))

    print(f"Optimized bids: {bids} ")
    print("final revenue:", bid_rev)

    u = np.zeros((n,T))

    for t in range(T):
        for i in range(n):
            u[i,t] = z[i,t]/(z[i,t]+y[i,t]) if (z[i,t]+y[i,t]) > 0 else 0
            if status[i][1] == 0:
                u[i,t] = 1 #force charge if no slack
            elif status[i][0] == 0:
                u[i,t] = 0 #force wait if no charge needed
    u = np.clip(u, 0.0, 1.0)

    os.makedirs("results", exist_ok=True)

    if eps is not None:
        path = f"results/SOC_B_policy_results_DP_eps{eps}_{scale_stats}_{DP}_{1-beta}.pkl" #
    else:
        path = f"results/SOC_B_policy_results_{int(scale_stats)}_{1-beta}.pkl" #{1-beta}

    print(f"Saving results to {path}")

    results = {
        "bids": bids,
        "bid_revenue": bid_rev,
        "status": status,
        "lambda": lam,
        "z": z,
        "y": y,
        "Mz": Mz_sol,
        "My": My_sol,
        "T": T,
    }
    with open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)