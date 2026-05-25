import math
import os
import tempfile
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

_MPL_DIR = os.path.join(tempfile.gettempdir(), "matplotlib")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm


# Shared SDT/Bayesian utilities used by all four architecture scripts.
# Keeping formulas here avoids copy-paste drift between files.
CUES: Tuple[str, str, str] = ("Sp", "Mp", "Np")


def std_norm_cdf(x: float) -> float:
    return float(norm.cdf(x))


def std_norm_sf(x: float) -> float:
    return float(norm.sf(x))


def default_base_params() -> Dict[str, float]:
    # Base case from the report.
    return {
        "Ps": 0.2,
        "dH": 2.5,
        "dAI": 2.5,
        "Blow": -2.0,
        "Bhigh": 2.0,
        "VTP": 100.0,
        "VTN": 100.0,
        "VFP": -100.0,
        "VFN": -100.0,
    }


def validate_params(params: Dict[str, float]) -> None:
    # Guardrails for parameter domains used by SDT equations.
    if not (0.0 < float(params["Ps"]) < 1.0):
        raise ValueError("Ps must be in (0, 1).")
    if float(params["dH"]) <= 0.0:
        raise ValueError("dH must be > 0.")
    if float(params["dAI"]) < 0.0:
        raise ValueError("dAI must be >= 0.")
    if float(params["Blow"]) >= float(params["Bhigh"]):
        raise ValueError("Require Blow < Bhigh.")
    if float(params["VTP"]) == float(params["VFN"]):
        raise ValueError("VTP - VFN must be non-zero.")


def ai_region_probabilities(params: Dict[str, float]) -> Dict[str, float]:
    # AI evidence regions are ordered exactly as:
    # Sp (right): x > Bhigh, Mp (middle): Blow < x < Bhigh, Np (left): x < Blow.
    dAI = float(params["dAI"])
    Blow = float(params["Blow"])
    Bhigh = float(params["Bhigh"])

    pNp_N = std_norm_cdf(Blow)
    pMp_N = std_norm_cdf(Bhigh) - std_norm_cdf(Blow)
    pSp_N = std_norm_sf(Bhigh)

    pNp_S = std_norm_cdf(Blow - dAI)
    pMp_S = std_norm_cdf(Bhigh - dAI) - std_norm_cdf(Blow - dAI)
    pSp_S = std_norm_sf(Bhigh - dAI)

    return {
        "pSp_S": pSp_S,
        "pMp_S": pMp_S,
        "pNp_S": pNp_S,
        "pSp_N": pSp_N,
        "pMp_N": pMp_N,
        "pNp_N": pNp_N,
    }


def payoff_ratio(params: Dict[str, float]) -> float:
    # Utility term that converts posterior odds into normative beta.
    VTP = float(params["VTP"])
    VTN = float(params["VTN"])
    VFP = float(params["VFP"])
    VFN = float(params["VFN"])
    ratio = (VTN - VFP) / (VTP - VFN)
    if ratio <= 0.0:
        raise ValueError("Payoff ratio (VTN-VFP)/(VTP-VFN) must be > 0 for log(beta).")
    return ratio


def posterior_signal_given_cue(
    Ps: float,
    pCue_S: float,
    pCue_N: float,
    eps: float = 1e-15,
) -> float:
    # Bayes update: P(S|cue) from prior and AI cue likelihoods.
    Pn = 1.0 - Ps
    p_cue = Ps * pCue_S + Pn * pCue_N
    if p_cue <= eps:
        return Ps
    return (Ps * pCue_S) / p_cue


def bstar_for_posterior(
    posterior_S: float,
    dH: float,
    payoff_ratio_value: float,
    eps: float = 1e-15,
) -> float:
    # Normative SDT threshold:
    # B* = dH/2 + ln(beta)/dH,  beta = (P(N|cue)/P(S|cue))*payoff_ratio.
    posterior_S_safe = min(max(posterior_S, eps), 1.0 - eps)
    posterior_N_safe = 1.0 - posterior_S_safe
    beta = (posterior_N_safe / posterior_S_safe) * payoff_ratio_value
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    return dH / 2.0 + math.log(beta) / dH


def human_sh_probabilities(dH: float, bstar: float) -> Tuple[float, float]:
    # Human chooses Sh when x_h > B*.
    pSh_S = std_norm_sf(bstar - dH)
    pSh_N = std_norm_sf(bstar)
    return pSh_S, pSh_N


def score_from_confusion(params: Dict[str, float], TP: float, TN: float, FP: float, FN: float) -> float:
    # Expected utility from confusion-matrix probabilities.
    return (
        TP * float(params["VTP"])
        + TN * float(params["VTN"])
        + FP * float(params["VFP"])
        + FN * float(params["VFN"])
    )


def validate_ai_probabilities(ai: Dict[str, float], tol: float = 1e-10) -> None:
    # Gaussian partition should naturally sum to 1 for each class.
    sum_S = ai["pSp_S"] + ai["pMp_S"] + ai["pNp_S"]
    sum_N = ai["pSp_N"] + ai["pMp_N"] + ai["pNp_N"]
    if abs(sum_S - 1.0) > tol:
        raise ValueError(f"P(Sp|S)+P(Mp|S)+P(Np|S) must be 1. Got {sum_S}.")
    if abs(sum_N - 1.0) > tol:
        raise ValueError(f"P(Sp|N)+P(Mp|N)+P(Np|N) must be 1. Got {sum_N}.")


def validate_confusion_total(TP: float, TN: float, FP: float, FN: float, tol: float = 1e-10) -> None:
    # System must always end in exactly one of TP/TN/FP/FN.
    total = TP + TN + FP + FN
    if abs(total - 1.0) > tol:
        raise ValueError(f"TP+TN+FP+FN must be 1. Got {total}.")


def validate_score_match(score_confusion: float, score_explicit: float, tol: float = 1e-9) -> None:
    # Ensures algebraic consistency between two score derivations.
    if abs(score_confusion - score_explicit) > tol:
        raise ValueError(
            f"Score mismatch: confusion={score_confusion}, explicit={score_explicit}."
        )


def _bounded_update(base_params: Dict[str, float], param_name: str, delta: float) -> Dict[str, float]:
    # Small bounded perturbation used for sensitivity sanity checks.
    params = dict(base_params)
    params[param_name] = float(params[param_name]) + delta

    if param_name == "Ps":
        params[param_name] = min(max(params[param_name], 0.01), 0.99)
    if param_name == "dH":
        params[param_name] = max(params[param_name], 0.1)
    if param_name == "dAI":
        params[param_name] = max(params[param_name], 0.0)
    if param_name == "Blow":
        params[param_name] = min(params[param_name], float(params["Bhigh"]) - 0.1)
    if param_name == "Bhigh":
        params[param_name] = max(params[param_name], float(params["Blow"]) + 0.1)
    return params


def validate_parameter_sensitivity(
    compute_outcomes: Callable[[Dict[str, float]], Dict[str, float]],
    base_params: Dict[str, float],
    tol: float = 1e-12,
) -> None:
    # Quick debug check: each key parameter should influence Score locally.
    base_score = compute_outcomes(base_params)["Score"]
    deltas = {
        "Ps": 0.03,
        "dH": 0.1,
        "dAI": 0.1,
        "Blow": -0.1,
        "Bhigh": 0.1,
    }
    for param_name, delta in deltas.items():
        params = _bounded_update(base_params, param_name, delta)
        alt_score = compute_outcomes(params)["Score"]
        if abs(alt_score - base_score) <= tol:
            raise ValueError(
                f"Parameter sensitivity check failed: changing {param_name} did not change Score."
            )


def build_sweep_values(sweep_min: float, sweep_max: float, sweep_step: float) -> np.ndarray:
    # Inclusive numeric grid builder for one-parameter sweeps.
    if sweep_step <= 0.0:
        raise ValueError("sweep_step must be positive.")
    values = np.arange(sweep_min, sweep_max + 0.5 * sweep_step, sweep_step)
    return np.round(values, 6)


def sweep_param(
    base_params: Dict[str, float],
    param_name: str,
    values: Iterable[float],
    compute_outcomes: Callable[[Dict[str, float]], Dict[str, float]],
) -> pd.DataFrame:
    # Evaluate one architecture over a sweep grid.
    rows = []
    for value in values:
        params = dict(base_params)
        params[param_name] = float(value)
        out = compute_outcomes(params)
        rows.append(
            {
                param_name: float(value),
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
    save_prefix: str,
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> None:
    # Minimal plotting helper used by all architecture scripts.
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


def report_sweep_ranges() -> Dict[str, np.ndarray]:
    # Sweep intervals defined in the report.
    return {
        "Blow": np.round(np.arange(-3.0, -1.0 + 1e-9, 0.1), 2),
        "Bhigh": np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2),
        "dAI": np.round(np.arange(1.5, 3.5 + 1e-9, 0.1), 2),
        "dH": np.round(np.arange(1.5, 3.5 + 1e-9, 0.1), 2),
        "Ps": np.round(np.arange(0.05, 0.95 + 1e-9, 0.02), 2),
    }
