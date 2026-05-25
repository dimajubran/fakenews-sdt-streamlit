import json
import subprocess
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st

# Streamlit dashboard for running and visualizing the four simulation architectures.
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
PLOT_WIDTH_PX = 780

# One tab per architecture script.
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
    "dH": {"min": 0.1, "max": 5.0, "default": 2.50, "step": 0.05},
    "dAI": {"min": 0.0, "max": 6.0, "default": 2.50, "step": 0.05},
    "Blow": {"min": -5.0, "max": 0.0, "default": -2.0, "step": 0.05},
    "Bhigh": {"min": 0.0, "max": 5.0, "default": 2.0, "step": 0.05},
}

# Default sweep ranges shown in the sidebar.
SWEEP_DEFAULTS = {
    "Ps": {"min": 0.05, "max": 0.95, "step": 0.05},
    "dH": {"min": 1.50, "max": 3.50, "step": 0.10},
    "dAI": {"min": 1.50, "max": 3.50, "step": 0.10},
    "Blow": {"min": -3.00, "max": -1.00, "step": 0.10},
    "Bhigh": {"min": 1.00, "max": 3.00, "step": 0.10},
}

BASE_PAYOFFS = {
    # Payoff names must match the refactored simulation scripts.
    "VTP": 100.0,
    "VTN": 100.0,
    "VFP": -100.0,
    "VFN": -100.0,
}


def _load_module(path: Path, module_name: str):
    # Dynamic import lets the app call compute_outcomes() from each architecture file.
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
    # Payload format expected by each architecture script's --params_json interface.
    return {
        "sweep_param": sweep_param,
        "sweep_min": float(sweep_min),
        "sweep_max": float(sweep_max),
        "sweep_step": float(sweep_step),
        "base_params": base_params,
    }


def _sort_pngs(paths: List[Path], sweep_param: str) -> List[Path]:
    # Prioritize the image matching the current sweep parameter.
    param_lower = sweep_param.lower()

    def _key(path: Path):
        name = path.name.lower()
        hit = param_lower in name
        return (0 if hit else 1, name)

    return sorted(paths, key=_key)


def _validate_params(
    base_params: Dict[str, float], sweep_min: float, sweep_max: float, sweep_step: float
) -> Optional[str]:
    # App-level validation before calling external scripts.
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
    # Compute current point metrics directly from Python API (in-memory),
    # alongside the separate sweep run that writes plots.
    params = dict(base_params)
    outcomes = module.compute_outcomes(params)
    # Extract whichever cue-specific B* values are returned by the active architecture.
    bstar_map = {
        key.replace("Bstar_", ""): value
        for key, value in outcomes.items()
        if key.startswith("Bstar_")
    }

    return {"outcomes": outcomes, "bstar_map": bstar_map}


def _run_simulation(script: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Executes the selected architecture script as a subprocess to generate sweep PNGs.
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
            # When sweep parameter changes, reset min/max/step to sensible defaults.
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
            # Disable direct editing of the swept parameter to avoid conflicting inputs.
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

        # Manual rerun knob for deterministic regeneration without changing parameters.
        if st.button("Recompute all tabs"):
            st.session_state["run_nonce"] = int(st.session_state.get("run_nonce", 0)) + 1

    return {
        "sweep_param": sweep_param,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "sweep_step": sweep_step,
        "base_params": base_params,
        "run_nonce": int(st.session_state.get("run_nonce", 0)),
    }


inputs = _render_sidebar()
tabs = st.tabs([cfg["label"] for cfg in TAB_CONFIGS])

def _inputs_signature(inputs: Dict[str, Any], base_params: Dict[str, float], cfg: Dict[str, str]) -> tuple:
    # Used for cache-like rerun control per tab.
    base_tuple = tuple((k, float(base_params[k])) for k in sorted(base_params.keys()))
    # Include code mtimes so code edits force recomputation.
    sdt_common_mtime = (ROOT_DIR / "sdt_common.py").stat().st_mtime_ns
    script_mtime = (ROOT_DIR / cfg["script"]).stat().st_mtime_ns
    return (
        inputs["sweep_param"],
        float(inputs["sweep_min"]),
        float(inputs["sweep_max"]),
        float(inputs["sweep_step"]),
        base_tuple,
        inputs["run_nonce"],
        sdt_common_mtime,
        script_mtime,
    )

for tab, cfg in zip(tabs, TAB_CONFIGS):
    with tab:
        st.title("Fake News Detection Simulation")
        base_params = {**inputs["base_params"], **BASE_PAYOFFS}

        state_key = f"state_{cfg['label']}"
        sig_key = f"sig_{cfg['label']}"

        st.markdown("## Results")
        # Only recompute when current parameter signature changed.
        error = _validate_params(
            base_params, inputs["sweep_min"], inputs["sweep_max"], inputs["sweep_step"]
        )
        if error:
            st.error(error)
        else:
            signature = _inputs_signature(inputs, base_params, cfg)
            if st.session_state.get(sig_key) != signature:
                payload = _build_payload(
                    inputs["sweep_param"],
                    inputs["sweep_min"],
                    inputs["sweep_max"],
                    inputs["sweep_step"],
                    base_params,
                )
                # 1) Run sweep script (produces plot files).
                run_result = _run_simulation(cfg["script"], payload)

                # 2) Compute single-point outcomes for metric cards.
                computed = None
                compute_error = None
                try:
                    module = _load_module(ROOT_DIR / cfg["script"], cfg["module_name"])
                    computed = _compute_results(module, cfg["label"], base_params)
                except Exception:
                    compute_error = traceback.format_exc()

                st.session_state[state_key] = {
                    "run": run_result,
                    "computed": computed,
                    "compute_error": compute_error,
                }
                st.session_state[sig_key] = signature

        state = st.session_state.get(state_key)
        outcomes = state["computed"]["outcomes"] if state and state.get("computed") else None
        bstar_map = state["computed"]["bstar_map"] if state and state.get("computed") else None

        # Top metric cards for quick inspection.
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
                    tpr_denominator = (tp + fn) if isinstance(tp, (int, float)) and isinstance(fn, (int, float)) else None
                    value = (tp / tpr_denominator) if tpr_denominator else None
                else:
                    fpr_denominator = (fp + tn) if isinstance(fp, (int, float)) and isinstance(tn, (int, float)) else None
                    value = (fp / fpr_denominator) if fpr_denominator else None
            else:
                value = outcomes.get(key) if outcomes else None
            display = f"{value:.4f}" if isinstance(value, (int, float)) else "—"
            col.markdown(label)
            col.markdown(display)

        # B* values shown for the cues that involve human decisions in each architecture.
        st.markdown("## Computed B* map")
        st.json(bstar_map if bstar_map is not None else {})

        # Optional deep-debug view for inspecting all computed fields.
        with st.expander("Debug details"):
            st.markdown("**Current payload**")
            st.json(
                _build_payload(
                    inputs["sweep_param"],
                    inputs["sweep_min"],
                    inputs["sweep_max"],
                    inputs["sweep_step"],
                    base_params,
                )
            )
            st.markdown("**Computed outcomes**")
            st.json(outcomes if outcomes is not None else {})
            if state:
                st.markdown("**Subprocess return code**")
                st.write(state["run"]["returncode"])
                if state["run"]["stderr"]:
                    st.markdown("**Subprocess stderr**")
                    st.code(state["run"]["stderr"], language="text")

        # Render the most relevant sweep image for the selected parameter.
        st.markdown("## Score vs Parameter Sweep")
        png_dir = ROOT_DIR / cfg["folder"]
        png_paths = list(png_dir.glob("*.png"))
        sorted_pngs = _sort_pngs(png_paths, inputs["sweep_param"])

        if sorted_pngs:
            st.image(str(sorted_pngs[0]), width=PLOT_WIDTH_PX)
        else:
            st.info("Run a simulation to generate plots.")

        if state and state["run"]["returncode"] != 0:
            st.error("Simulation failed. Check the terminal logs for details.")
            if state["run"]["stderr"]:
                st.code(state["run"]["stderr"], language="text")
        if state and state.get("compute_error"):
            st.error("Live metric computation failed:")
            st.code(state["compute_error"], language="text")


#streamlit run simulationapp.py
