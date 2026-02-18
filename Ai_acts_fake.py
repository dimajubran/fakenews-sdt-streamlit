import os
import math
import argparse
import json
from typing import Optional, Dict

# Limit math libraries to a single thread for stability on constrained systems.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

def std_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return norm.cdf(x)


def std_norm_sf(x: np.ndarray) -> np.ndarray:
    """Standard normal survival function (1 - CDF)."""
    return norm.sf(x)


def _validate_params(params: Dict[str, float]) -> None:
    """Sanity-check the main parameters."""
    if not (0 < params["Ps"] < 1):
        raise ValueError("Ps must be in (0,1).")
    if params["dH"] <= 0 or params["dAI"] < 0:
        raise ValueError("d′ must be positive.")
    if params["Blow"] >= params["Bhigh"]:
        raise ValueError("Require Blow < Bhigh.")


def compute_outcomes(params: Dict[str, float]) -> Dict[str, float]:
    """
    AI acts only on FAKE: AI makes the final call on S' (right region), while the
    human decides for N' and M' with cue-specific thresholds computed each call.
    """
    _validate_params(params)
    Ps = float(params["Ps"])
    Pn = 1.0 - Ps
    dH = float(params["dH"])
    dAI = float(params["dAI"])
    Blow = float(params["Blow"])
    Bhigh = float(params["Bhigh"])
    V_TP = float(params["V_TP"])
    V_TN = float(params["V_TN"])
    V_FP = float(params["V_FP"])
    V_FN = float(params["V_FN"])

    # --- AI region probabilities (not renormalized) ---
    P_Nprime_given_N = std_norm_cdf(Blow)
    P_Mprime_given_N = std_norm_cdf(Bhigh) - std_norm_cdf(Blow)
    P_Sprime_given_N = std_norm_sf(Bhigh)

    P_Nprime_given_S = std_norm_cdf(Blow - dAI)
    P_Mprime_given_S = std_norm_cdf(Bhigh - dAI) - std_norm_cdf(Blow - dAI)
    P_Sprime_given_S = std_norm_sf(Bhigh - dAI)

    # --- Human queue posterior Ps_h (items with x_AI < Bhigh) ---
    P_H_given_N = std_norm_cdf(Bhigh)
    P_H_given_S = std_norm_cdf(Bhigh - dAI)
    denom_h = Ps * P_H_given_S + Pn * P_H_given_N
    denom_h = max(denom_h, 1e-12)
    Ps_h = (Ps * P_H_given_S) / denom_h

    # --- Cue-specific posterior probabilities ---
    eps = 1e-12
    denom_N = Ps * P_Nprime_given_S + Pn * P_Nprime_given_N
    denom_M = Ps * P_Mprime_given_S + Pn * P_Mprime_given_N
    denom_N = max(denom_N, eps)
    denom_M = max(denom_M, eps)

    P_S_given_Nprime = (Ps * P_Nprime_given_S) / denom_N
    P_S_given_Mprime = (Ps * P_Mprime_given_S) / denom_M
    P_S_given_Nprime = float(np.clip(P_S_given_Nprime, eps, 1.0 - eps))
    P_S_given_Mprime = float(np.clip(P_S_given_Mprime, eps, 1.0 - eps))
    P_N_given_Nprime = 1.0 - P_S_given_Nprime
    P_N_given_Mprime = 1.0 - P_S_given_Mprime

    payoff_ratio = (V_TN - V_FP) / (V_TP - V_FN)
    beta_N = (P_N_given_Nprime / P_S_given_Nprime) * payoff_ratio
    beta_M = (P_N_given_Mprime / P_S_given_Mprime) * payoff_ratio
    beta_N = max(beta_N, eps)
    beta_M = max(beta_M, eps)

    Bstar_N = (dH / 2.0) + (math.log(beta_N) / dH)
    Bstar_M = (dH / 2.0) + (math.log(beta_M) / dH)

    # --- Human response probabilities per cue ---
    p_h_fake_given_S_N = std_norm_sf(Bstar_N - dH)
    p_h_fake_given_N_N = std_norm_sf(Bstar_N)
    p_h_fake_given_S_M = std_norm_sf(Bstar_M - dH)
    p_h_fake_given_N_M = std_norm_sf(Bstar_M)

    # --- Aggregate confusion matrix ---
    TP_AI = Ps * P_Sprime_given_S
    FP_AI = Pn * P_Sprime_given_N

    TP_H = Ps * (
        P_Nprime_given_S * p_h_fake_given_S_N
        + P_Mprime_given_S * p_h_fake_given_S_M
    )
    FN_H = Ps * (
        P_Nprime_given_S * (1.0 - p_h_fake_given_S_N)
        + P_Mprime_given_S * (1.0 - p_h_fake_given_S_M)
    )
    FP_H = Pn * (
        P_Nprime_given_N * p_h_fake_given_N_N
        + P_Mprime_given_N * p_h_fake_given_N_M
    )
    TN_H = Pn * (
        P_Nprime_given_N * (1.0 - p_h_fake_given_N_N)
        + P_Mprime_given_N * (1.0 - p_h_fake_given_N_M)
    )

    TP = TP_AI + TP_H
    FP = FP_AI + FP_H
    TN = TN_H
    FN = FN_H

    Score = TP * V_TP + TN * V_TN + FP * V_FP + FN * V_FN

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Score": Score,
        "Ps_h": Ps_h,
        "Bstar_N": Bstar_N,
        "Bstar_M": Bstar_M,
        "P_S_given_Nprime": P_S_given_Nprime,
        "P_S_given_Mprime": P_S_given_Mprime,
    }


def sweep_param(base_params: Dict[str, float], param_name: str, values: np.ndarray) -> pd.DataFrame:
    """Sweep one parameter while holding the rest fixed."""
    rows = []
    for v in values:
        params = dict(base_params)
        params[param_name] = float(v)
        out = compute_outcomes(params)
        rows.append(
            {
                param_name: float(v),
                "Score": out["Score"],
                "TP": out["TP"],
                "TN": out["TN"],
                "FP": out["FP"],
                "FN": out["FN"],
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(
    df: pd.DataFrame,
    param_name: str,
    title: str,
    figsize=(8, 5),
    save_prefix: str = "Ai_acts_fake",
) -> None:
    """Plot Score vs parameter with consistent styling and a fixed Score axis."""
    plt.figure(figsize=figsize)
    plt.plot(df[param_name], df["Score"], marker="o", linestyle="-")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.title(title)

    base_step = 0.1 if param_name != "Ps" else 0.05
    x_min, x_max = df[param_name].min(), df[param_name].max()
    tick_start = np.floor(x_min / base_step) * base_step
    tick_end = np.ceil(x_max / base_step) * base_step + 1e-12
    ticks = np.arange(tick_start, tick_end + 0.5 * base_step, base_step)
    plt.xticks(ticks)

    plt.tight_layout()
    out_path = f"{save_prefix}_{param_name}_sweep.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def _default_base_params() -> Dict[str, float]:
    return {
        "Ps": 0.2,
        "dH": 2.5,
        "dAI": 2.5,
        "Blow": -2.0,
        "Bhigh": 2.0,
        "V_TP": 100.0,
        "V_TN": 100.0,
        "V_FP": -100.0,
        "V_FN": -100.0,
    }


def _build_sweep_values(sweep_min: float, sweep_max: float, sweep_step: float) -> np.ndarray:
    if sweep_step <= 0:
        raise ValueError("sweep_step must be positive.")
    values = np.arange(sweep_min, sweep_max + 0.5 * sweep_step, sweep_step)
    return np.round(values, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI-acts-fake simulation sweeps.")
    parser.add_argument("--params_json", type=str, default=None)
    args = parser.parse_args()

    out_dir = "Ai acts fake"
    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, "Ai_acts_fake")

    base_params = _default_base_params()

    if args.params_json:
        payload = json.loads(args.params_json)
        base_params.update(payload.get("base_params", {}))
        sweep_param_name = payload.get("sweep_param")
        sweep_min = float(payload.get("sweep_min"))
        sweep_max = float(payload.get("sweep_max"))
        sweep_step = float(payload.get("sweep_step"))

        values = _build_sweep_values(sweep_min, sweep_max, sweep_step)
        df = sweep_param(base_params, sweep_param_name, values)
        plot_sweep(
            df,
            sweep_param_name,
            f"Score vs {sweep_param_name} ",
            save_prefix=save_prefix,
        )
        raise SystemExit(0)

    Blow_values = np.round(np.arange(-3.0, -1.0 + 1e-9, 0.1), 2)
    Bhigh_values = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
    d_values = np.round(np.arange(1.0, 3.5 + 1e-9, 0.1), 2)
    Ps_values = np.round(np.arange(0.02, 0.99 + 1e-9, 0.02), 2)

    df_blow = sweep_param(base_params, "Blow", Blow_values)
    df_bhigh = sweep_param(base_params, "Bhigh", Bhigh_values)
    df_dai = sweep_param(base_params, "dAI", d_values)
    df_dh = sweep_param(base_params, "dH", d_values)
    df_ps = sweep_param(base_params, "Ps", Ps_values)

    plot_sweep(df_blow, "Blow", "Score vs Blow ", save_prefix=save_prefix)
    plot_sweep(df_bhigh, "Bhigh", "Score vs Bhigh ", save_prefix=save_prefix)
    plot_sweep(df_dai, "dAI", "Score vs d' AI ", save_prefix=save_prefix)
    plot_sweep(df_dh, "dH", "Score vs d' Human ", save_prefix=save_prefix)
    plot_sweep(df_ps, "Ps", "Score vs Ps ", save_prefix=save_prefix)
