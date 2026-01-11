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

# Order of AI recommendations (Fake, Middle, Real) used throughout.
R_ORDER = ["F", "M", "R"]
# Precompute sqrt(2) for Gaussian CDF/SF calculations with math.erf/erfc.  שורש של 2
SQRT2 = np.sqrt(2.0) # שורש של 2


def std_norm_cdf(x: np.ndarray) -> np.ndarray: 
    """Standard normal CDF using math.erf applied elementwise."""
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / SQRT2))


def std_norm_sf(x: np.ndarray) -> np.ndarray:
    """Standard normal survival function (1 - CDF) via math.erfc."""
    erfc_vec = np.vectorize(math.erfc)
    return 0.5 * erfc_vec(x / SQRT2)


def _validate_params(params: Dict[str, float]) -> None: #-> None: the function returns nothing
    """Sanity-check the main parameters."""
    if not (0 < params["Ps"] < 1):  #0<P(S)<1
        raise ValueError("Ps must be in (0,1).")
    if params["dH"] <= 0 or params["dAI"] < 0:  #dH>0, dAI>=0
        raise ValueError("d′ must be positive.")
    if params["Blow"] >= params["Bhigh"]:      #Blow<Bhigh
        raise ValueError("Require Blow < Bhigh.")


def _stabilize_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip probabilities away from 0/1 and renormalize (numerical hygiene)."""
    clipped = np.clip(probs, eps, 1 - eps) 
    return clipped / clipped.sum()


def compute_outcomes(params: Dict[str, float]) -> Dict[str, float]:
    """
    Forward pass of the hybrid model. AI emits F/M/R via two thresholds; the human
    always decides last using cue-specific thresholds stored in 'Bstar_map'.
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
    if "Bstar_map" not in params:
        raise KeyError("params must include 'Bstar_map' with thresholds per AI cue.")
    k_map = params["Bstar_map"] #B* conditioned on AI recommendation

    # --- AI recommendation probabilities (equal-variance SDT) ---
    p_r_given_N_raw = np.array(
        [
            1 - std_norm_cdf(Bhigh),                 # AI says Fake under Noise
            std_norm_cdf(Bhigh) - std_norm_cdf(Blow),# AI says Middle
            std_norm_cdf(Blow),                      # AI says Real
        ]
    )
    p_r_given_S_raw = np.array(
        [
            1 - std_norm_cdf(Bhigh - dAI),                 # AI says Fake under Signal
            std_norm_cdf(Bhigh - dAI) - std_norm_cdf(Blow - dAI),
            std_norm_cdf(Blow - dAI),
        ]
    )
    p_r_given_N = _stabilize_probs(p_r_given_N_raw)
    p_r_given_S = _stabilize_probs(p_r_given_S_raw)

    # --- Human probabilities: use the precomputed cue-specific thresholds k(r) ---
    k_arr = np.array([k_map[r] for r in R_ORDER], dtype=float) #k_map={"F": k_F,"M": k_M,"R": k_R}
    p_h_fake_given_S = std_norm_sf(k_arr - dH)  # human fake decision when item is signal
    p_h_fake_given_N = std_norm_sf(k_arr)       # human fake decision when item is noise

    # --- Aggregate to confusion-matrix probabilities ---
    TP = Ps * np.dot(p_r_given_S, p_h_fake_given_S)
    FN = Ps * np.dot(p_r_given_S, 1.0 - p_h_fake_given_S)
    FP = Pn * np.dot(p_r_given_N, p_h_fake_given_N)
    TN = Pn * np.dot(p_r_given_N, 1.0 - p_h_fake_given_N)

    # --- Expected payoff score ---
    Score = TP * V_TP + TN * V_TN + FP * V_FP + FN * V_FN

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "Score": Score}


def sweep_param(base_params: Dict[str, float], param_name: str, values: np.ndarray) -> pd.DataFrame:
    """Sweep one parameter while holding the rest fixed."""
    rows = []
    for v in values:
        params = dict(base_params)
        params[param_name] = float(v)
        # Recompute cue-specific human thresholds when Ps changes.
        if param_name == "Ps":
            params["Bstar_map"] = compute_bstar_map(params)
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
    save_prefix: str = "simulation",
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


def compute_bstar_map(ref_params: Dict[str, float]) -> Dict[str, float]:
    """
    Compute cue-specific human thresholds k(r) once, using Amit's beta* derivation.
    They remain fixed during sweeps so the human behavior is tied to the base case.
    """
    _validate_params(ref_params)
    Ps = float(ref_params["Ps"])
    Pn = 1.0 - Ps
    dH = float(ref_params["dH"])
    dAI = float(ref_params["dAI"])
    Blow = float(ref_params["Blow"])
    Bhigh = float(ref_params["Bhigh"])
    V_TP = float(ref_params["V_TP"])
    V_TN = float(ref_params["V_TN"])
    V_FP = float(ref_params["V_FP"])
    V_FN = float(ref_params["V_FN"])

    # AI recommendation probabilities for the base case.
    p_r_given_N = _stabilize_probs(
        np.array(
            [
                1 - std_norm_cdf(Bhigh),
                std_norm_cdf(Bhigh) - std_norm_cdf(Blow),
                std_norm_cdf(Blow),
            ]
        )
    )
    p_r_given_S = _stabilize_probs(
        np.array(
            [
                1 - std_norm_cdf(Bhigh - dAI),
                std_norm_cdf(Bhigh - dAI) - std_norm_cdf(Blow - dAI),
                std_norm_cdf(Blow - dAI),
            ]
        )
    )

    # Posterior P(S|r) for the base case and resulting beta*(r).
    denom = Ps * p_r_given_S + Pn * p_r_given_N
    denom = np.clip(denom, 1e-12, None)
    P_S_given_r = np.clip((Ps * p_r_given_S) / denom, 1e-12, 1 - 1e-12)
    P_N_given_r = 1.0 - P_S_given_r
    payoff_ratio = (V_TN - V_FP) / (V_TP - V_FN)
    beta_r = np.clip((P_N_given_r / P_S_given_r) * payoff_ratio, 1e-12, None)

    # Convert beta into the effective decision threshold on the evidence axis.
    C_r = np.log(beta_r) / dH
    k_r = (dH / 2.0) + C_r
    return dict(zip(R_ORDER, k_r))


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
    parser = argparse.ArgumentParser(description="Run AI-recommends-only simulation sweeps.")
    parser.add_argument("--params_json", type=str, default=None)
    args = parser.parse_args()

    out_dir = "Ai recommends only"
    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, "simulation")

    # Base-case parameters used for both the computation and for deriving Bstar_map.
    base_params = _default_base_params()

    if args.params_json:
        payload = json.loads(args.params_json)
        base_params.update(payload.get("base_params", {}))
        base_params["Bstar_map"] = compute_bstar_map(base_params)
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

    # Human thresholds (per AI cue) are tied to the base scenario.
    base_params["Bstar_map"] = compute_bstar_map(base_params)

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

    # Generate and save plots with fixed Score axis limits.
    plot_sweep(df_blow, "Blow", "Score vs Blow ", save_prefix=save_prefix)
    plot_sweep(df_bhigh, "Bhigh", "Score vs Bhigh ", save_prefix=save_prefix)
    plot_sweep(df_dai, "dAI", "Score vs d' AI ", save_prefix=save_prefix)
    plot_sweep(df_dh, "dH", "Score vs d' Human ", save_prefix=save_prefix)
    plot_sweep(df_ps, "Ps", "Score vs Ps ", save_prefix=save_prefix)
