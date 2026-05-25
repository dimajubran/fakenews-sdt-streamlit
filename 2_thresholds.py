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
    Architecture 2: Two-Thresholds.
    AI autonomously decides Sp->Sh and Np->Nh. Human decides only Mp.
    """
    # 1) Validate inputs and compute AI likelihoods per region.
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

    # 2) Human only sees Mp in this architecture.
    #    Therefore only Bstar_Mp is needed.
    ratio = payoff_ratio(params)
    posterior_Mp = posterior_signal_given_cue(Ps, ai["pMp_S"], ai["pMp_N"])
    Bstar_Mp = bstar_for_posterior(posterior_Mp, dH, ratio)
    pSh_S_Mp, pSh_N_Mp = human_sh_probabilities(dH, Bstar_Mp)

    # 3) Routing:
    #    Sp -> AI final Sh, Np -> AI final Nh, Mp -> human decision.
    TP = Ps * (ai["pSp_S"] + ai["pMp_S"] * pSh_S_Mp)
    FN = Ps * (ai["pNp_S"] + ai["pMp_S"] * (1.0 - pSh_S_Mp))
    FP = Pn * (ai["pSp_N"] + ai["pMp_N"] * pSh_N_Mp)
    TN = Pn * (ai["pNp_N"] + ai["pMp_N"] * (1.0 - pSh_N_Mp))

    validate_confusion_total(TP, TN, FP, FN)

    # 4) Validate score equivalence against explicit branch formula.
    score_confusion = score_from_confusion(params, TP, TN, FP, FN)
    score_explicit = (
        Ps
        * (
            ai["pSp_S"] * VTP
            + ai["pMp_S"] * (pSh_S_Mp * VTP + (1.0 - pSh_S_Mp) * VFN)
            + ai["pNp_S"] * VFN
        )
        + Pn
        * (
            ai["pSp_N"] * VFP
            + ai["pMp_N"] * (pSh_N_Mp * VFP + (1.0 - pSh_N_Mp) * VTN)
            + ai["pNp_N"] * VTN
        )
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
        "P_S_given_Mp": posterior_Mp,
        "Bstar_Mp": Bstar_Mp,
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
    parser = argparse.ArgumentParser(description="Run Two-Thresholds simulation sweeps.")
    parser.add_argument("--params_json", type=str, default=None)
    args = parser.parse_args()

    out_dir = "2 thresholds"
    os.makedirs(out_dir, exist_ok=True)
    save_prefix = os.path.join(out_dir, "2_thresholds")

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
