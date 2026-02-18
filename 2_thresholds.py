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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Precompute sqrt(2) for Gaussian CDF/SF calculations with math.erf/erfc.
SQRT2 = np.sqrt(2.0)


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
    Forward pass for AI-ACTS (2 thresholds): AI decides on the sides, the human
    decides only on the middle without any additional AI cue.
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
    pS_N = 1.0 - std_norm_cdf(Bhigh)  # P(S'|N)

    pN_S = std_norm_cdf(Blow - dAI)  # P(N'|S)
    pM_S = std_norm_cdf(Bhigh - dAI) - std_norm_cdf(Blow - dAI)  # P(M'|S)
    pS_S = 1.0 - std_norm_cdf(Bhigh - dAI)  # P(S'|S)

    # --- Human posterior given the middle region only ---
    eps = 1e-12
    denom = Ps * pM_S + Pn * pM_N
    denom = max(denom, eps)
    Ps_h = (Ps * pM_S) / denom
    Ps_h = float(np.clip(Ps_h, eps, 1.0 - eps))

    # --- Single Bayes-optimal human threshold ---
    payoff_ratio = (V_TN - V_FP) / (V_TP - V_FN)
    beta_h = ((1.0 - Ps_h) / Ps_h) * payoff_ratio #
    beta_h = max(beta_h, eps)
    B_h_star = dH / 2.0 + math.log(beta_h) / dH 

    p_h_fake_given_S = float(std_norm_sf(np.array([B_h_star - dH]))[0])
    p_h_fake_given_N = float(std_norm_sf(np.array([B_h_star]))[0])

    # --- Aggregate confusion-matrix probabilities ---
    TP_AI = Ps * pS_S
    FP_AI = Pn * pS_N
    TN_AI = Pn * pN_N
    FN_AI = Ps * pN_S

    TP_H = Ps * pM_S * p_h_fake_given_S
    FN_H = Ps * pM_S * (1.0 - p_h_fake_given_S)
    FP_H = Pn * pM_N * p_h_fake_given_N
    TN_H = Pn * pM_N * (1.0 - p_h_fake_given_N)

    TP = TP_AI + TP_H
    FP = FP_AI + FP_H
    TN = TN_AI + TN_H
    FN = FN_AI + FN_H
    # print(Ps, pS_S, TN_AI)

    Score = Ps * (pS_S * V_TP + pM_S * (p_h_fake_given_S * V_TP + (1 - p_h_fake_given_S) * V_FN) + pN_S * V_FN) + Pn * (pS_N * V_FP + pM_N * (p_h_fake_given_N * V_FP + (1 - p_h_fake_given_N) * V_TN) + pN_N * V_TN)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Score": Score,
        "Ps_h": Ps_h,
        "B_h_star": B_h_star,
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
    save_prefix: str = "2_thresholds",
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
        "Ps": 0.5,
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
    parser = argparse.ArgumentParser(description="Run 2-thresholds simulation sweeps.")
    parser.add_argument("--params_json", type=str, default=None)
    args = parser.parse_args()

    out_dir = "2 thresholds"
    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, "2_thresholds")

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
    #print(std_norm_sf(np.array([0])))
