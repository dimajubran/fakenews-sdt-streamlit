import argparse
import json
import os
from typing import Dict

# Limit math libraries to a single thread for stability on constrained systems.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from sdt_common import (
    ai_region_probabilities,
    bstar_for_posterior,
    build_sweep_values,
    default_base_params,
    human_sh_probabilities,
    payoff_ratio,
    plot_sweep,
    posterior_signal_given_cue,
    report_sweep_ranges,
    score_from_confusion,
    sweep_param,
    validate_ai_probabilities,
    validate_confusion_total,
    validate_parameter_sensitivity,
    validate_params,
    validate_score_match,
)


def compute_outcomes(params: Dict[str, float]) -> Dict[str, float]:
    """
    Architecture 1: AI-Recommends-Only.
    AI provides Sp/Mp/Np for every item. Human always makes the final Sh/Nh decision.
    """
    # 1) Validate parameter ranges and compute AI region likelihoods.
    validate_params(params)
    ai = ai_region_probabilities(params)
    validate_ai_probabilities(ai)

    Ps = float(params["Ps"])
    Pn = 1.0 - Ps
    dH = float(params["dH"])
    VTP = float(params["VTP"])
    VTN = float(params["VTN"])
    VFP = float(params["VFP"])
    VFN = float(params["VFN"])

    # 2) Convert payoffs to the SDT payoff ratio term.
    ratio = payoff_ratio(params)

    # 3) For AI-Recommends-Only, human sees all three cues.
    #    So we compute cue-conditioned posteriors and B* for Sp/Mp/Np separately.
    posteriors = {}
    bstars = {}
    pSh_S = {}
    pSh_N = {}
    for cue in ("Sp", "Mp", "Np"):
        pCue_S = ai[f"p{cue}_S"]
        pCue_N = ai[f"p{cue}_N"]
        posterior_S = posterior_signal_given_cue(Ps, pCue_S, pCue_N)
        bstar = bstar_for_posterior(posterior_S, dH, ratio)
        p_fake_given_S, p_fake_given_N = human_sh_probabilities(dH, bstar)

        posteriors[cue] = posterior_S
        bstars[cue] = bstar
        pSh_S[cue] = p_fake_given_S
        pSh_N[cue] = p_fake_given_N

    # 4) Aggregate branch outcomes into TP/TN/FP/FN.
    TP = Ps * sum(ai[f"p{cue}_S"] * pSh_S[cue] for cue in ("Sp", "Mp", "Np"))
    FN = Ps * sum(ai[f"p{cue}_S"] * (1.0 - pSh_S[cue]) for cue in ("Sp", "Mp", "Np"))
    FP = Pn * sum(ai[f"p{cue}_N"] * pSh_N[cue] for cue in ("Sp", "Mp", "Np"))
    TN = Pn * sum(ai[f"p{cue}_N"] * (1.0 - pSh_N[cue]) for cue in ("Sp", "Mp", "Np"))

    validate_confusion_total(TP, TN, FP, FN)

    # 5) Score is checked in two equivalent ways for safety.
    score_confusion = score_from_confusion(params, TP, TN, FP, FN)
    score_explicit = Ps * sum(
        ai[f"p{cue}_S"] * (pSh_S[cue] * VTP + (1.0 - pSh_S[cue]) * VFN)
        for cue in ("Sp", "Mp", "Np")
    ) + Pn * sum(
        ai[f"p{cue}_N"] * (pSh_N[cue] * VFP + (1.0 - pSh_N[cue]) * VTN)
        for cue in ("Sp", "Mp", "Np")
    )
    validate_score_match(score_confusion, score_explicit)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Score": score_confusion,
        "pSp_S": ai["pSp_S"],
        "pMp_S": ai["pMp_S"],
        "pNp_S": ai["pNp_S"],
        "pSp_N": ai["pSp_N"],
        "pMp_N": ai["pMp_N"],
        "pNp_N": ai["pNp_N"],
        "P_S_given_Sp": posteriors["Sp"],
        "P_S_given_Mp": posteriors["Mp"],
        "P_S_given_Np": posteriors["Np"],
        "Bstar_Sp": bstars["Sp"],
        "Bstar_Mp": bstars["Mp"],
        "Bstar_Np": bstars["Np"],
    }


def _run_single_sweep(base_params: Dict[str, float], payload: Dict[str, float], save_prefix: str) -> None:
    # UI/API path: run one requested sweep.
    sweep_param_name = payload["sweep_param"]
    values = build_sweep_values(
        float(payload["sweep_min"]),
        float(payload["sweep_max"]),
        float(payload["sweep_step"]),
    )
    df = sweep_param(base_params, sweep_param_name, values, compute_outcomes)
    plot_sweep(df, sweep_param_name, f"Score vs {sweep_param_name}", save_prefix=save_prefix)


def _run_default_sweeps(base_params: Dict[str, float], save_prefix: str) -> None:
    # CLI path: generate all standard report sweeps.
    for param_name, values in report_sweep_ranges().items():
        df = sweep_param(base_params, param_name, values, compute_outcomes)
        title_map = {
            "Blow": "Score vs Blow",
            "Bhigh": "Score vs Bhigh",
            "dAI": "Score vs dAI",
            "dH": "Score vs dH",
            "Ps": "Score vs Ps",
        }
        plot_sweep(df, param_name, title_map[param_name], save_prefix=save_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI-Recommends-Only simulation sweeps.")
    parser.add_argument("--params_json", type=str, default=None)
    args = parser.parse_args()

    out_dir = "Ai recommends only"
    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, "simulation")

    # Base values are report-consistent unless caller overrides with params_json.
    base_params = default_base_params()
    if args.params_json:
        payload = json.loads(args.params_json)
        base_params.update(payload.get("base_params", {}))
        validate_parameter_sensitivity(compute_outcomes, base_params)
        _run_single_sweep(base_params, payload, save_prefix)
    else:
        validate_parameter_sensitivity(compute_outcomes, base_params)
        _run_default_sweeps(base_params, save_prefix)
