import json
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
      .stTabs [data-baseweb="tab"] p {
        font-size: 1.1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parent

TAB_CONFIGS = [
    {
        "label": "2 thresholds",
        "script": "2_thresholds.py",
        "folder": "2 thresholds",
        "module_name": "sim_2_thresholds",
    },
    {
        "label": "AI acts fake",
        "script": "Ai_acts_fake.py",
        "folder": "Ai acts fake",
        "module_name": "sim_ai_acts_fake",
    },
    {
        "label": "AI acts real",
        "script": "Ai_acts_real.py",
        "folder": "Ai acts real",
        "module_name": "sim_ai_acts_real",
    },
    {
        "label": "AI recommends only",
        "script": "simulation.py",
        "folder": "Ai recommends only",
        "module_name": "sim_ai_recommends_only",
    },
]

PARAM_RANGES = {
    "Ps": {"min": 0.05, "max": 0.95, "default": 0.20, "step": 0.01},
    "dH": {"min": 0.0, "max": 5.0, "default": 2.50, "step": 0.05},
    "dAI": {"min": 0.0, "max": 6.0, "default": 2.50, "step": 0.05},
    "Blow": {"min": -5.0, "max": 0.0, "default": -2.0, "step": 0.05},
    "Bhigh": {"min": 0.0, "max": 5.0, "default": 2.0, "step": 0.05},
}

SWEEP_DEFAULTS = {
    "Ps": {"min": 0.05, "max": 0.95, "step": 0.05},
    "dH": {"min": 1.50, "max": 3.50, "step": 0.10},
    "dAI": {"min": 1.50, "max": 3.50, "step": 0.10},
    "Blow": {"min": -3.00, "max": -1.00, "step": 0.10},
    "Bhigh": {"min": 1.00, "max": 3.00, "step": 0.10},
}

BASE_PAYOFFS = {
    "V_TP": 100.0,
    "V_TN": 100.0,
    "V_FP": -100.0,
    "V_FN": -100.0,
}


@st.cache_resource
def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_payload(
    sweep_param: str,
    sweep_min: float,
    sweep_max: float,
    sweep_step: float,
    base_params: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "sweep_param": sweep_param,
        "sweep_min": float(sweep_min),
        "sweep_max": float(sweep_max),
        "sweep_step": float(sweep_step),
        "base_params": base_params,
    }


def _sort_pngs(paths: List[Path], sweep_param: str) -> List[Path]:
    param_lower = sweep_param.lower()

    def _key(path: Path):
        name = path.name.lower()
        hit = param_lower in name
        return (0 if hit else 1, name)

    return sorted(paths, key=_key)


def _validate_params(
    base_params: Dict[str, float], sweep_min: float, sweep_max: float, sweep_step: float
) -> Optional[str]:
    if not (0 < base_params["Ps"] < 1):
        return "Ps must be between 0 and 1."
    if base_params["dH"] <= 0 or base_params["dAI"] < 0:
        return "dH must be > 0 and dAI must be >= 0."
    if base_params["Blow"] >= base_params["Bhigh"]:
        return "Blow must be less than Bhigh."
    if sweep_min >= sweep_max:
        return "Sweep range min must be less than max."
    if sweep_step <= 0:
        return "Sweep step must be positive."
    return None


def _compute_results(module, tab_label: str, base_params: Dict[str, float]) -> Dict[str, Any]:
    params = dict(base_params)
    bstar_map = None
    if tab_label == "AI recommends only":
        params["Bstar_map"] = module.compute_bstar_map(params)
        bstar_map = params["Bstar_map"]
    outcomes = module.compute_outcomes(params)

    if tab_label == "2 thresholds":
        bstar_map = {"M": outcomes.get("B_h_star")}
    elif tab_label == "AI acts fake":
        bstar_map = {
            "M": outcomes.get("Bstar_M"),
            "R": outcomes.get("Bstar_N"),
        }
    elif tab_label == "AI acts real":
        bstar_map = {
            "F": outcomes.get("Bstar_S"),
            "M": outcomes.get("Bstar_M"),
        }

    return {"outcomes": outcomes, "bstar_map": bstar_map}


def _run_simulation(script: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cmd = [
        "python3",
        str(ROOT_DIR / script),
        "--params_json",
        json.dumps(payload),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### Parameter Sweep")

        def _set_sweep_defaults() -> None:
            param = st.session_state.get("sweep_param", list(PARAM_RANGES.keys())[0])
            cfg = SWEEP_DEFAULTS[param]
            st.session_state["sweep_min"] = float(cfg["min"])
            st.session_state["sweep_max"] = float(cfg["max"])
            st.session_state["sweep_step"] = float(cfg["step"])

        sweep_param = st.selectbox(
            "Sweep parameter",
            list(PARAM_RANGES.keys()),
            key="sweep_param",
            on_change=_set_sweep_defaults,
        )
        param_range = PARAM_RANGES[sweep_param]
        sweep_defaults = SWEEP_DEFAULTS[sweep_param]
        sweep_min = st.number_input(
            "Sweep range (min)",
            min_value=float(param_range["min"]),
            max_value=float(param_range["max"]) - float(param_range["step"]),
            value=float(sweep_defaults["min"]),
            step=float(sweep_defaults["step"]),
            format="%.3f",
            key="sweep_min",
        )
        sweep_max = st.number_input(
            "Sweep range (max)",
            min_value=float(param_range["min"]) + float(param_range["step"]),
            max_value=float(param_range["max"]),
            value=float(sweep_defaults["max"]),
            step=float(sweep_defaults["step"]),
            format="%.3f",
            key="sweep_max",
        )
        sweep_step = st.number_input(
            "Sweep step",
            min_value=0.001,
            value=float(sweep_defaults["step"]),
            step=float(sweep_defaults["step"]),
            format="%.3f",
            key="sweep_step",
        )

        st.markdown("### Base Case Parameters")
        base_params = {}
        for name, cfg in PARAM_RANGES.items():
            if name == "Ps":
                base_params[name] = st.number_input(
                    name,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    format="%.3f",
                    disabled=(name == sweep_param),
                    key=f"base_{name}",
                )
            else:
                base_params[name] = st.number_input(
                    name,
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    format="%.3f",
                    disabled=(name == sweep_param),
                    key=f"base_{name}",
                )

    return {
        "sweep_param": sweep_param,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "sweep_step": sweep_step,
        "base_params": base_params,
    }


inputs = _render_sidebar()
tabs = st.tabs([cfg["label"] for cfg in TAB_CONFIGS])

def _inputs_signature(inputs: Dict[str, Any], base_params: Dict[str, float]) -> tuple:
    base_tuple = tuple((k, float(base_params[k])) for k in sorted(base_params.keys()))
    return (
        inputs["sweep_param"],
        float(inputs["sweep_min"]),
        float(inputs["sweep_max"]),
        float(inputs["sweep_step"]),
        base_tuple,
    )

for tab, cfg in zip(tabs, TAB_CONFIGS):
    with tab:
        st.title("Fake News Detection Simulation")
        base_params = {**inputs["base_params"], **BASE_PAYOFFS}

        state_key = f"state_{cfg['label']}"
        sig_key = f"sig_{cfg['label']}"

        st.markdown("## Results")
        error = _validate_params(
            base_params, inputs["sweep_min"], inputs["sweep_max"], inputs["sweep_step"]
        )
        if error:
            st.error(error)
        else:
            signature = _inputs_signature(inputs, base_params)
            if st.session_state.get(sig_key) != signature:
                payload = _build_payload(
                    inputs["sweep_param"],
                    inputs["sweep_min"],
                    inputs["sweep_max"],
                    inputs["sweep_step"],
                    base_params,
                )
                run_result = _run_simulation(cfg["script"], payload)

                module = _load_module(ROOT_DIR / cfg["script"], cfg["module_name"])
                computed = _compute_results(module, cfg["label"], base_params)

                st.session_state[state_key] = {
                    "run": run_result,
                    "computed": computed,
                }
                st.session_state[sig_key] = signature

        state = st.session_state.get(state_key)
        outcomes = state["computed"]["outcomes"] if state else None
        bstar_map = state["computed"]["bstar_map"] if state else None

        metrics_cols = st.columns(7)
        metrics = [
            ("Score", "Score"),
            ("TP", r"$P_{TP}$"),
            ("TN", r"$P_{TN}$"),
            ("FP", r"$P_{FP}$"),
            ("FN", r"$P_{FN}$"),
            ("TPR", r"$\frac{TP}{TP+FN}$"),
            ("FPR", r"$\frac{FP}{FP+TN}$"),
        ]
        for col, (key, label) in zip(metrics_cols, metrics):
            if outcomes and key in ("TPR", "FPR"):
                tp = outcomes.get("TP")
                fn = outcomes.get("FN")
                fp = outcomes.get("FP")
                tn = outcomes.get("TN")
                if key == "TPR":
                    denom = (tp + fn) if isinstance(tp, (int, float)) and isinstance(fn, (int, float)) else None
                    value = (tp / denom) if denom else None
                else:
                    denom = (fp + tn) if isinstance(fp, (int, float)) and isinstance(tn, (int, float)) else None
                    value = (fp / denom) if denom else None
            else:
                value = outcomes.get(key) if outcomes else None
            display = f"{value:.4f}" if isinstance(value, (int, float)) else "—"
            col.markdown(label)
            col.markdown(display)

        st.markdown("## Computed B* map")
        st.json(bstar_map if bstar_map is not None else {})

        st.markdown("## Score vs Parameter Sweep")
        png_dir = ROOT_DIR / cfg["folder"]
        png_paths = list(png_dir.glob("*.png"))
        sorted_pngs = _sort_pngs(png_paths, inputs["sweep_param"])

        if sorted_pngs:
            st.image(str(sorted_pngs[0]), use_container_width=True)
        else:
            st.info("Run a simulation to generate plots.")

        if state and state["run"]["returncode"] != 0:
            st.error("Simulation failed. Check the terminal logs for details.")
