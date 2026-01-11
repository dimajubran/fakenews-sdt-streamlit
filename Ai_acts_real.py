import os
import math
import argparse
import json
from typing import Dict

# Limit math libraries to a single thread for stability on constrained systems.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Precompute sqrt(2) for Gaussian CDF/SF calculations with math.erf/erfc.  שורש של 2
SQRT2 = np.sqrt(2.0)  # שורש של 2


def std_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using math.erf applied elementwise."""
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / SQRT2))


def std_norm_sf(x: np.ndarray) -> np.ndarray:
    """Standard normal survival function (1 - CDF) via math.erfc."""
    erfc_vec = np.vectorize(math.erfc)
    return 0.5 * erfc_vec(x / SQRT2)


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
    Forward pass for AI acts only on REAL: AI decides final on N' (left),
    the human decides on M' and S' with cue-specific thresholds.
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

    # --- AI region probabilities (no renormalization) ---
    pN_N = std_norm_cdf(Blow)  # P(N'|N)
    pM_N = std_norm_cdf(Bhigh) - std_norm_cdf(Blow)  # P(M'|N)
    pS_N = std_norm_sf(np.array([Bhigh]))[0]  # P(S'|N)

    pN_S = std_norm_cdf(Blow - dAI)  # P(N'|S)
    pM_S = std_norm_cdf(Bhigh - dAI) - std_norm_cdf(Blow - dAI)  # P(M'|S)
    pS_S = std_norm_sf(np.array([Bhigh - dAI]))[0]  # P(S'|S)

    # --- Ps_h for overall human queue H = {x_AI > Blow} ---
    eps = 1e-12
    pH_N = std_norm_sf(np.array([Blow]))[0]
    pH_S = std_norm_sf(np.array([Blow - dAI]))[0]
    denom_h = Ps * pH_S + Pn * pH_N
    denom_h = max(denom_h, eps)
    Ps_h = (Ps * pH_S) / denom_h
    Ps_h = float(np.clip(Ps_h, eps, 1.0 - eps))

    # --- Cue-specific posteriors for M' and S' ---
    denom_m = Ps * pM_S + Pn * pM_N
    denom_s = Ps * pS_S + Pn * pS_N
    denom_m = max(denom_m, eps)
    denom_s = max(denom_s, eps)
    P_S_given_M = (Ps * pM_S) / denom_m
    P_S_given_S = (Ps * pS_S) / denom_s
    P_S_given_M = float(np.clip(P_S_given_M, eps, 1.0 - eps))
    P_S_given_S = float(np.clip(P_S_given_S, eps, 1.0 - eps))
    P_N_given_M = 1.0 - P_S_given_M
    P_N_given_S = 1.0 - P_S_given_S

    payoff_ratio = (V_TN - V_FP) / (V_TP - V_FN)
    beta_M = (P_N_given_M / P_S_given_M) * payoff_ratio
    beta_S = (P_N_given_S / P_S_given_S) * payoff_ratio
    beta_M = max(beta_M, eps)
    beta_S = max(beta_S, eps)

    Bstar_M = dH / 2.0 + math.log(beta_M) / dH
    Bstar_S = dH / 2.0 + math.log(beta_S) / dH

    p_h_fake_given_S_M = float(std_norm_sf(np.array([Bstar_M - dH]))[0])
    p_h_fake_given_N_M = float(std_norm_sf(np.array([Bstar_M]))[0])
    p_h_fake_given_S_S = float(std_norm_sf(np.array([Bstar_S - dH]))[0])
    p_h_fake_given_N_S = float(std_norm_sf(np.array([Bstar_S]))[0])

    # --- Aggregate confusion-matrix probabilities ---
    TN_AI = Pn * pN_N
    FN_AI = Ps * pN_S

    TP_H = Ps * (pM_S * p_h_fake_given_S_M + pS_S * p_h_fake_given_S_S)
    FN_H = Ps * (
        pM_S * (1.0 - p_h_fake_given_S_M) + pS_S * (1.0 - p_h_fake_given_S_S)
    )
    FP_H = Pn * (pM_N * p_h_fake_given_N_M + pS_N * p_h_fake_given_N_S)
    TN_H = Pn * (
        pM_N * (1.0 - p_h_fake_given_N_M) + pS_N * (1.0 - p_h_fake_given_N_S)
    )

    TP = TP_H
    FP = FP_H
    TN = TN_AI + TN_H
    FN = FN_AI + FN_H

    Score = TP * V_TP + TN * V_TN + FP * V_FP + FN * V_FN

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Score": Score,
        "Ps_h": Ps_h,
        "Bstar_M": Bstar_M,
        "Bstar_S": Bstar_S,
        "P_S_given_M": P_S_given_M,
        "P_S_given_S": P_S_given_S,
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
    save_prefix: str = "Ai_acts_real",
) -> None:
    """Plot Score vs parameter with consistent styling and a fixed Score axis."""
    plt.figure(figsize=figsize)
    plt.plot(df[param_name], df["Score"], marker="o", linestyle="-")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.title(title)
    # Let matplotlib auto-scale the y-axis for this variant.

    # Use a step of 0.1 for most parameters; Ps is labeled every 0.05 for clarity.
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
    parser = argparse.ArgumentParser(description="Run AI-acts-real simulation sweeps.")
    parser.add_argument("--params_json", type=str, default=None)
    args = parser.parse_args()

    out_dir = "Ai acts real"
    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, "Ai_acts_real")

    # Base-case parameters used for both the computation and sweeps.
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

    # Sweep grids for each parameter (others stay fixed at base values).
    Blow_values = np.round(np.arange(-3.0, -1.0 + 1e-9, 0.1), 2)
    Bhigh_values = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
    d_values = np.round(np.arange(1.0, 3.5 + 1e-9, 0.1), 2)
    Ps_values = np.round(np.arange(0.02, 0.99 + 1e-9, 0.02), 2)

    # Run sweeps.
    df_blow = sweep_param(base_params, "Blow", Blow_values)
    df_bhigh = sweep_param(base_params, "Bhigh", Bhigh_values)
    df_dai = sweep_param(base_params, "dAI", d_values)
    df_dh = sweep_param(base_params, "dH", d_values)
    df_ps = sweep_param(base_params, "Ps", Ps_values)

    # Generate and save plots.
    plot_sweep(df_blow, "Blow", "Score vs Blow ", save_prefix=save_prefix)
    plot_sweep(df_bhigh, "Bhigh", "Score vs Bhigh ", save_prefix=save_prefix)
    plot_sweep(df_dai, "dAI", "Score vs d' AI ", save_prefix=save_prefix)
    plot_sweep(df_dh, "dH", "Score vs d' Human ", save_prefix=save_prefix)
    plot_sweep(df_ps, "Ps", "Score vs Ps ", save_prefix=save_prefix)
