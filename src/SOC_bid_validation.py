import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

from collections import defaultdict

from lam_price_setting import read_daily_prices

@dataclass
class ParsedPolicyOutput:
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
    Mz: Optional[Dict[Tuple[int,int,int,int], float]] = None
    My: Optional[Dict[Tuple[int,int,int,int], float]] = None


def load_record_arrivals(file_path: Path, status: List[Tuple[int, int]], time_labels: List[int]) -> List[np.ndarray]:
    """Load an EV record CSV and aggregate it into one arrival matrix per day."""

    df = pd.read_csv(file_path)
    required_columns = {"day", "time_step", "state_idx", "status_charge","status_slack"}

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {sorted(missing)}")

    day_range = df["day"].unique() if not df.empty else []
    arrivals_by_day: List[np.ndarray] = []

    for day in day_range:
        day_df = df[df["day"] == day]
        arrivals = np.zeros((len(status), len(time_labels)), dtype=float)
        for (status_charge, status_slack,time_step), group in day_df.groupby(["status_charge", "status_slack","time_step"]):
            state_idx = next((i for i, (c, s) in enumerate(status) if c == status_charge and s == status_slack), None)
            if state_idx is not None and 0 <= time_step < len(time_labels):
                arrivals[state_idx, time_step] = float(len(group))
      
        arrivals_by_day.append(arrivals)

    return arrivals_by_day


def parse_policy_output(file_path: Path) -> ParsedPolicyOutput:
    """Parse the CSV-like output written by `bidOptimizer_policy.py`."""

    with open(file_path, "rb") as f:
        results = pickle.load(f)

    # ------------------------
    # basic data
    # ------------------------
    bids = results["bids"]
    bid_revenue = results["bid_revenue"]

    status_order = results["status"]

    lam = np.asarray(results["lambda"])

    Mz_entries = results["Mz"]
    My_entries = results["My"]

    z_bar = results["z"]
    y_bar = results["y"]

    T = results["T"]

    # ------------------------
    # bids
    # ------------------------
    if len(bids) == 0:
        raise ValueError(f"No bids found in {file_path}")

    hour_labels = np.arange(len(bids))

    bid_up = np.array([b[0] for b in bids], dtype=float)
    bid_down = np.array([b[1] for b in bids], dtype=float)

    # old parser expected these,
    # but they are no longer stored per hour
    objective_value = bid_revenue
    reliability = 0

    # ------------------------
    # time labels
    # ------------------------
    time_labels = np.arange(lam.shape[1])

    return ParsedPolicyOutput(
        bid_up=bid_up,
        bid_down=bid_down,
        objective_value=objective_value,
        reliability=reliability,
        lam=lam,
        status=status_order,
        hour_labels=hour_labels,
        time_labels=time_labels,
        Mz=Mz_entries,
        My=My_entries,
        z=z_bar,
        y=y_bar
    )

def build_hour_to_steps(time_labels: List[int], hour_count: int) -> Dict[int, List[int]]:
    """Map hour indices to the time-step indices belonging to each hour."""
    if hour_count <= 0:
        raise ValueError("hour_count must be positive")
    step_count = len(time_labels)
    steps_per_hour = max(1, int(round(step_count / hour_count)))
    hour_to_steps: Dict[int, List[int]] = {h: [] for h in range(hour_count)}

    for t_idx in range(step_count):
        hour = min(hour_count - 1, t_idx // steps_per_hour)
        hour_to_steps[hour].append(t_idx)
    return hour_to_steps

def build_transition_sets(status: List[Tuple[int, int]]) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int], List[int]]]:
    """Build the same one-step transition lookup used in the optimization code."""
    next_charge: Dict[Tuple[int, int], List[int]] = {}
    next_slack: Dict[Tuple[int, int], List[int]] = {}

    for idx, (charge, slack) in enumerate(status):
        next_charge.setdefault((charge - 1, slack), []).append(idx)
        next_slack.setdefault((charge, slack - 1), []).append(idx)

    return next_charge, next_slack

def reachable_states(arrival_state, tau):
    """
    Compute all states reachable from an arrival state after `tau`
    time steps.

    Parameters
    ----------
    arrival_state : tuple[int, int]
        Initial state represented as (charge, slack).

    tau : int
        Number of elapsed time steps.

    Returns
    -------
    list[tuple[int, int]]
        Reachable states after `tau` transitions.
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
    Construct the active state-transition index set used by the
    recourse policy representation.

    Parameters
    ----------
    status : list[tuple[int, int]]
        State space represented as (charge, slack) pairs.

    lam : np.ndarray
        Expected arrival-rate matrix of shape (n_states, T).

    T : int
        Number of time steps.

    Returns
    -------
    Mactive : list[tuple[int, int, int, int]]
        Active transition tuples (t, k, j, i).

    Reach : dict
        Mapping (t, k, j) -> reachable destination states.
    """
    state_to_index = {s: i for i, s in enumerate(status)}

    Mactive = []
    Reach = {}

    for k in range(T):
        # only arrival states that actually have mass at time k
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
                    # boundary states are deterministic (mandatory charge / wait)
                    # and are not tracked as independent M-variables
                    if c == 0 or s == 0:
                        continue

                    i_dest = state_to_index[x]
                    reach.append(i_dest)
                    Mactive.append((t, k, j, i_dest))

                Reach[(t, k, j)] = reach

    return Mactive, Reach


def simulate_single_path(
        arrivals,
        lam,
        status,
        z,
        y,
        Mz,
        My,
        dt,
        p_rate,
        el_price, p_bup, p_bdown):
    """
    Simulate the EV fleet evolution for a single arrival realization.
    Parameters
    ----------
    arrivals : np.ndarray
        Realized EV arrivals for the scenario.

    lam : np.ndarray
        Expected arrival-rate matrix used during optimization.

    status : list[tuple[int, int]]
        State definitions as (charge, slack) pairs.

    z, y : np.ndarray
        Baseline charging and waiting occupancies.

    Mz, My : dict
        Recourse policy mappings for charging and waiting states.

    dt : float
        Duration of a simulation time step [h].

    p_rate : float
        Charging power per EV [kW].

    el_price : np.ndarray
        Electricity price trajectory.

    p_bup : np.ndarray
        Historical upward reserve prices used to complete missing policies.

    p_bdown : np.ndarray
        Historical downward reserve prices used to complete missing policies.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Upward reserve capacity, downward reserve capacity,
        and total electricity cost.
    """

    n_states, T = arrivals.shape

    next_charge, next_slack = build_transition_sets(status)

    reserve_mask = np.array(
        [c > 0 and s > 0 for c, s in status],
        dtype=bool
    )

    r_up = np.zeros(T)
    r_down = np.zeros(T)

    el_cost = 0.0

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

    z_bar = np.zeros((n_states, T))
    y_bar = np.zeros((n_states, T))

    n_current = lam[:, 0].astype(float).copy()
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
            n_current = B @ n_current + lam[:, t + 1]

    for t in range(T):
        for i in range(len(status)):
            if np.abs(z_bar[i, t]-z[i, t]) > 1e-1 or np.abs(y_bar[i, t]-y[i, t]) > 1e-1:
                print(f"Different z,y_bar t={t}, i={i}, z_bar={z_bar[i,t]},z {z[i,t]} y_bar={y_bar[i,t]} y {y[i,t]}")

    Mz_by_tk = defaultdict(list)
    My_by_tk = defaultdict(list)

    Mactive, _ = build_active_states(status, lam, T)
    active_arrivals = [
        (k, j) for k in range(T) for j in range(len(status)) if lam[j, k] > 0
    ]

    for (t, k, j, i), val in Mz.items():
        Mz_by_tk[(t, k)].append((j, i, val))

    for (t, k, j, i), val in My.items():
        My_by_tk[(t, k)].append((j, i, val))

    mandatory_charge_states = {
        i for i, (c, s) in enumerate(status)
        if c > 0 and s == 0
    }

    full_charge_states = {
        i for i, (c, s) in enumerate(status)
        if c == 0 and s > 0
    }
    for (t,k,j,i) in Mactive:
        if (t,k,j,i) not in Mz and (t,k,j,i) not in My:
            print(f"Active state {t,k,j,i} has no Mz or My entry")
        if i in mandatory_charge_states and (t,k,j,i) in My:
            if My[t,k,j,i] != 0.0:
                print(f"Mandatory charge state {t,k,j,i} has nonzero My: {My[t,k,j,i]}")
        if i in full_charge_states and (t,k,j,i) in Mz:
            if Mz[t,k,j,i] != 0.0:
                print(f"Full charge state {t,k,j,i} has nonzero Mz: {Mz[t,k,j,i]}")

    delta_a = arrivals - lam 

    missing_policy_entries = [(j,k) for j in range(len(status)) for k in range(T) if arrivals[j,k] > 0 and (k,j) not in active_arrivals]
    Mz_miss = {}
    My_miss = {}
    for j,k in missing_policy_entries:
        i = j
        for t in range(k, T):
            if (t,k,j,i) not in Mz and (t,k,j,i) not in My:
                if status[i][0] > 0:
                    i_ch =  status.index((status[i][0]-1, status[i][1])) 
                else:
                    i_ch = None
                if status[i][1] > 0:
                    i_wait = status.index((status[i][0], status[i][1]-1)) 
                else:
                    i_wait = None
                

                if p_bup[int(t*dt)] > 5 * p_bdown[int(t*dt)]:
                   if i_wait is not None:
                       My_miss[(t,k,j,i_wait)] = 1
                       i = i_wait
                       if i_ch is not None:
                           Mz_miss[(t,k,j,i_ch)] = 0
                   else:
                       if i_ch is not None:
                           Mz_miss[(t,k,j,i_ch)] = 1
                       i = i_ch
                else:
                    if i_ch is not None:
                        Mz_miss[(t,k,j,i_ch)] = 1.0
                        if i_wait is not None:
                            My_miss[(t,k,j,i_wait)] = 0
                        i = i_ch
                    else:
                        if i_wait is not None:
                            My_miss[(t,k,j,i_wait)] = 1
                        i = i_wait
                if i is None:
                    #print(f"No next state for {status[j]} at time {t}, skipping further updates")
                    break

    for (t, k, j, i), val in Mz_miss.items():
        Mz_by_tk[(t, k)].append((j, i, val))

    for (t, k, j, i), val in My_miss.items():
        My_by_tk[(t, k)].append((j, i, val))


    neg_z=[]
    neg_y=[]

    for t in range(T):

        z_real = z_bar[:, t].copy()
        y_real = y_bar[:, t].copy()

        for k in range(t+1):

            disturbance = delta_a[:, k]
   
            for j,i,val in Mz_by_tk.get((t,k), ()):
                z_real[i] += val * disturbance[j]


            for j,i,val in My_by_tk.get((t,k), ()):
                y_real[i] += val * disturbance[j]


        neg_z.append(np.sum(z_real < -1e-1))
        neg_y.append(np.sum(y_real < -1e-1))
        if neg_z[-1] > 0 :
            if np.min(z_real[z_real < -1e-1]) < -1:
                print(f"Time step {t}: negative z: {neg_z[-1]} minimum value: {np.min(z_real[z_real < -1e-1]) if z_real[z_real < -1e-1].size > 0 else np.inf}" )
        if neg_y[-1] > 0:
            if np.min(y_real[y_real < -1e-1]) < -1:
                print(f"Time step {t}: negative y: {neg_y[-1]} minimum value: {np.min(y_real[y_real < -1e-1]) if y_real[y_real < -1e-1].size > 0 else np.inf}" )

        z_real = np.maximum(z_real, 0)
        y_real = np.maximum(y_real, 0)

        r_up[t] = np.sum(z_real[reserve_mask])
        r_down[t] = np.sum(y_real[reserve_mask])

        el_cost += (
            np.sum(z_real)
            * el_price[t]
            * dt
        )

    return (
        r_up * p_rate,
        r_down * p_rate,
        el_cost,
    )

def validate_bids(parsed: ParsedPolicyOutput, samples: int, seed: int, dt: float, p_rate: float,
                  arrival_paths: Optional[List[np.ndarray]] = None):
    """Run validation and return empirical feasibility rates.
    Parameters
    ----------
    parsed : ParsedPolicyOutput
        The optimized policy to validate.

    samples : int
        The number of Monte Carlo samples to generate.

    seed : int
        The random seed for reproducibility.

    dt : float
        The duration of a simulation time step [h].

    p_rate : float
        The charging power per EV [kW].

    arrival_paths : Optional[List[np.ndarray]]
        A list of pre-generated arrival paths. If None, new paths are generated.
    """
    hour_count = len(parsed.bid_up)
    hour_to_steps = build_hour_to_steps(parsed.time_labels, hour_count)
    
    rng = np.random.default_rng(seed)
    
    # -----------------------------
    # Energinet statistics
    # -----------------------------
    total_valid_samples = 0
    total_success_samples = 0

    per_hour_success = np.zeros(hour_count, dtype=int)
    per_hour_total = np.zeros(hour_count, dtype=int)
    per_hour_min_margin = np.full(hour_count, np.inf, dtype=float)

    overall_success = 0

    if arrival_paths is None:
        arrival_paths = [rng.poisson(parsed.lam) for _ in range(samples)]
    else:
        samples = len(arrival_paths)

    if not arrival_paths:
        raise ValueError("No arrival paths available for validation")


    net_profit = np.zeros(samples, dtype=float)
    net_profit_for_EV = np.zeros(samples, dtype=float)
    margin_up_samples = np.zeros((samples, hour_count))
    margin_down_samples = np.zeros((samples, hour_count))

    daily_prices = read_daily_prices("data/fcrd_price.csv","data/spot_price.csv")

    # Stack hourly profiles
    price_el_d = [np.repeat(daily_prices[d]["spot"], int(1/dt)) for d in range(60,60+samples)]

    price_bup_d = [daily_prices[d]["fcrd_down"] for d in range(60,60+samples)]

    price_bdown_d = [daily_prices[d]["fcrd_up"] for d in range(60,60+samples)]

    price_bup_d_is = [daily_prices[d]["fcrd_down"] for d in range(60)]
    price_bdown_d_is = [daily_prices[d]["fcrd_up"] for d in range(60)]

    past_price_bup = np.mean(price_bup_d_is, axis=0)
    past_price_bdown = np.mean(price_bdown_d_is, axis=0)

    cap_ups = np.zeros((samples, hour_count), dtype=float)
    cap_downs = np.zeros((samples, hour_count), dtype=float)
 
    for s,arrivals in enumerate(arrival_paths):
        c = [i for i, (charge, slack) in enumerate(parsed.status) if charge > 0 and slack > 0]
        
        r_up, r_down , el_cost = simulate_single_path(
                                    arrivals=arrivals,
                                    lam=parsed.lam,
                                    status=parsed.status,
                                    z = parsed.z,
                                    y = parsed.y,
                                    Mz=parsed.Mz,
                                    My=parsed.My,
                                    dt=dt,
                                    p_rate=p_rate,
                                    el_price=price_el_d[s],
                                    p_bup=past_price_bup, #default policy price for missing policy entries
                                    p_bdown=past_price_bdown #default policy price for missing policy entries
                                )
        
        r_up = np.maximum(r_up, 0)
        r_down = np.maximum(r_down, 0)


        sample_profit = 0.0
        sample_all_hours_ok = True
        for h in range(hour_count):
            steps = hour_to_steps[h]
            if not steps:
                continue

            cap_up = float(np.min(r_up[steps]))
            cap_down = float(np.min(r_down[steps]))

            cap_ups[s, h] = cap_up
            cap_downs[s, h] = cap_down

            bid_up = float(parsed.bid_up[h])
            bid_down = float(parsed.bid_down[h])

            # Calculate the revenue from the bid
            bid_revenue = bid_up * price_bup_d[s][h] + bid_down * price_bdown_d[s][h] # make the price also dependent on the day realization

            margin_up = cap_up - (bid_up + 0.2 * bid_down)
            margin_down = cap_down - bid_down
            
            per_hour_min_margin[h] = min(per_hour_min_margin[h], margin_up, margin_down)
            margin_up_samples[s, h] = margin_up
            margin_down_samples[s, h] = margin_down

            
            is_feasible = (margin_up >= -1e-1) and (margin_down >= -1e-1)

            # Count per-hour stats
            per_hour_total[h] += 1
            if is_feasible:
                per_hour_success[h] += 1

            # ✅ GLOBAL P90 statistic
            total_valid_samples += 1
            if is_feasible:
                total_success_samples += 1

            if not is_feasible:
                sample_all_hours_ok = False

            sample_profit += bid_revenue

        if sample_all_hours_ok:
            overall_success += 1

        net_profit[s] = sample_profit - el_cost
        net_profit_for_EV[s] = (sample_profit - el_cost) / np.sum(arrivals[c,:])# net profit for EV owners
    
    # Energinet P90 metric
    p_hat = total_success_samples / total_valid_samples

    # Standard error (for plots!)
    p_hat_std = np.sqrt(p_hat * (1 - p_hat) / total_valid_samples)

    # Per-hour stats
    empirical_hour_prob = per_hour_success / np.maximum(per_hour_total, 1)
    empirical_hour_prob_std = np.sqrt(
        empirical_hour_prob * (1 - empirical_hour_prob) / np.maximum(per_hour_total, 1)
    )   

    return {
        "p90_success_rate": p_hat,
        "p90_success_rate_std": p_hat_std,

        "total_valid_samples": total_valid_samples,

        # Per-hour stats
        "empirical_hour_prob": empirical_hour_prob,
        "empirical_hour_prob_std": empirical_hour_prob_std,

        # Conservative test (too strong but useful)
        "empirical_overall_prob": overall_success / float(samples),

        # Margins
        "margin_up_samples": margin_up_samples,
        "margin_down_samples": margin_down_samples,

        "per_hour_min_margin": per_hour_min_margin,

        # Profit
        "net_profit_mean": float(np.mean(net_profit)),
        "net_profit_std": float(np.std(net_profit, ddof=1)/np.sqrt(samples)),  # Standard error of the mean
        "net_profit_all": net_profit,

        "net_profit_for_EV_mean": float(np.mean(net_profit_for_EV)),
        "net_profit_for_EV_std": float(np.std(net_profit_for_EV, ddof=1)/np.sqrt(samples)),  # Standard error of the mean
        "net_profit_for_EV_all": net_profit_for_EV,
    }




def main() -> int:
    """
    Run Monte Carlo validation of optimized reserve bids.

    The script loads an optimized policy, generates or replays arrival
    scenarios, evaluates reserve-delivery feasibility, and saves both
    summary statistics and margin distributions.

    Returns
    -------
    int
        Exit status code.
    """
    parser = argparse.ArgumentParser(description="Monte Carlo bid feasibility validation.")
    parser.add_argument("--input", type=Path, default=Path("results/CalCC_opt_policy_results.csv"), help="CSV produced by bidOptimizer_policy.py")
    parser.add_argument("--samples", type=int, default=10000, help="Number of Poisson arrival realizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--scale", type=float, default=100.0, help="Scaling factor for the lambda table (for sensitivity analysis)")
    parser.add_argument("--p-rate", type=float, default=12.0, help="Charging power rate used in the constraints")
    parser.add_argument("--dt", type=float, default=1, help="Time step. If omitted, inferred from T/H")
    parser.add_argument("--output", type=Path, default=Path("results/CalCC_mc_bidvalidation_summary_records.csv"), help="Where to save the summary CSV")
    parser.add_argument("--eps", type=float, default=None, help="Epsilon value for lambda inference (used if control reconstruction is attempted)")
    parser.add_argument("--records", type=Path, default=None, help="Optional EV record CSV to replay instead of sampling Poisson arrivals")
    args = parser.parse_args()

    if args.eps is not None:
        args.output = args.output.with_name(args.output.stem + f"_eps{args.eps}_{int(args.scale)}" + args.output.suffix)
        args.input = args.input.with_name(args.input.stem + f"_DP_eps{args.eps}_{int(args.scale)}" + args.input.suffix)

    parsed = parse_policy_output(args.input)
    hour_count = len(parsed.bid_up)
    time_count = len(parsed.time_labels)

    
    if args.dt is None:
        if hour_count == 0:
            raise ValueError("Cannot infer dt from an empty bid list")
        dt = hour_count / float(time_count)
    else:
        dt = args.dt

    
    arrival_paths = None
    if args.records is not None:
        arrival_paths = load_record_arrivals(args.records, parsed.status, parsed.time_labels)

    result = validate_bids(parsed, samples=args.samples, seed=args.seed, dt=dt, p_rate=args.p_rate, arrival_paths=arrival_paths)

    print(f"Parsed file: {args.input}")
    print(f"Hours: {hour_count}, time steps: {time_count}, dt: {dt:.4f}, samples: {args.samples}")
    print(f"Empirical overall feasibility: {result['empirical_overall_prob']:.4f}")
    print("Per-hour feasibility:")
    for h in range(hour_count):
        print(
            f"  Hour {h}: bid_up={parsed.bid_up[h]:.4f}, bid_down={parsed.bid_down[h]:.4f}, "
            f"P(feasible)={result['empirical_hour_prob'][h]:.4f}, min margin={result['per_hour_min_margin'][h]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("hour,bid_up,bid_down,objective_value,reliability,empirical_feasibility,min_margin,net_profit_mean,net_profit_std,net_profit_for_EV_mean,net_profit_for_EV_std,p90_success_rate,p90_success_rate_std\n")         
        for h in range(hour_count):
            f.write(
                f"{h},{parsed.bid_up[h]:.6f},{parsed.bid_down[h]:.6f},{parsed.objective_value},"
                f"{parsed.reliability},{result['empirical_hour_prob'][h]:.6f},{result['per_hour_min_margin'][h]:.6f},"
                f"{result['net_profit_mean']:.6f},{result['net_profit_std']:.6f},"
                f"{result['net_profit_for_EV_mean']:.6f},{result['net_profit_for_EV_std']:.6f},"
                f"{result['p90_success_rate']},{result['p90_success_rate_std']}\n"
            )
    
    # Save full distributions separately
    np.savetxt(
        args.output.with_name(args.output.stem + "_margin_up.csv"),
        result["margin_up_samples"],
        delimiter=","
    )

    np.savetxt(
        args.output.with_name(args.output.stem + "_margin_down.csv"),
        result["margin_down_samples"],
        delimiter=","
    )


    print(f"Summary written to: {args.output}")
    print(f"Margin up samples written to: {args.output.with_name(args.output.stem + '_margin_up.csv')}")
    print(f"Margin down samples written to: {args.output.with_name(args.output.stem + '_margin_down.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())