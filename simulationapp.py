import json
import subprocess
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sdt_common import fake_real_distribution_across_ai_regions


st.set_page_config(layout="wide")

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
    "dH": {"min": 0.1, "max": 5.0, "default": 2.50, "step": 0.05},
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
    "VTP": 100.0,
    "VTN": 100.0,
    "VFP": -100.0,
    "VFN": -100.0,
}


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
    base_params: Dict[str, float],
    sweep_min: float,
    sweep_max: float,
    sweep_step: float,
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
    outcomes = module.compute_outcomes(params)

    bstar_map = {
        key.replace("Bstar_", ""): value
        for key, value in outcomes.items()
        if key.startswith("Bstar_")
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


def _inputs_signature(
    inputs: Dict[str, Any],
    base_params: Dict[str, float],
    cfg: Dict[str, str],
) -> tuple:
    base_tuple = tuple((k, float(base_params[k])) for k in sorted(base_params.keys()))

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


def normal_pdf(x, mean, std=1):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def plot_sdt_ai_regions(Ps: float, Blow: float, Bhigh: float, dAI: float):
    """
    Plot the SDT AI score distributions and show the 3 AI threshold regions.

    Assumptions:
    - Real/noise distribution mean = -dAI / 2
    - Fake/signal distribution mean = +dAI / 2
    - Both distributions have standard deviation = 1
    - Ps affects the height of the fake curve
    - 1 - Ps affects the height of the real curve
    """

    mu_real = -dAI / 2
    mu_fake = dAI / 2

    x_min = min(mu_real - 4, Blow - 1)
    x_max = max(mu_fake + 4, Bhigh + 1)

    x = np.linspace(x_min, x_max, 1000)

    real_pdf = (1 - Ps) * normal_pdf(x, mean=mu_real, std=1)
    fake_pdf = Ps * normal_pdf(x, mean=mu_fake, std=1)

    y_max = max(real_pdf.max(), fake_pdf.max())

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(x, real_pdf, label=f"Real items / Noise, μ = {mu_real:.2f}")
    ax.plot(x, fake_pdf, label=f"Fake items / Signal, μ = {mu_fake:.2f}")

    ax.axvline(Blow, linestyle="--", linewidth=2, label=f"Blow = {Blow:.2f}")
    ax.axvline(Bhigh, linestyle="--", linewidth=2, label=f"Bhigh = {Bhigh:.2f}")

    ax.fill_between(
        x,
        0,
        np.maximum(real_pdf, fake_pdf),
        where=(x < Blow),
        alpha=0.10,
    )

    ax.fill_between(
        x,
        0,
        np.maximum(real_pdf, fake_pdf),
        where=((x >= Blow) & (x <= Bhigh)),
        alpha=0.10,
    )

    ax.fill_between(
        x,
        0,
        np.maximum(real_pdf, fake_pdf),
        where=(x > Bhigh),
        alpha=0.10,
    )

    ax.text(
        (x_min + Blow) / 2,
        y_max * 0.92,
        "Lower\nx < Blow",
        ha="center",
        va="center",
    )

    ax.text(
        (Blow + Bhigh) / 2,
        y_max * 0.92,
        "Middle\nBlow < x < Bhigh",
        ha="center",
        va="center",
    )

    ax.text(
        (Bhigh + x_max) / 2,
        y_max * 0.92,
        "Upper\nx > Bhigh",
        ha="center",
        va="center",
    )

    ax.set_title("SDT AI score distributions and AI threshold regions")
    ax.set_xlabel("AI internal evidence / score")
    ax.set_ylabel("Weighted density")
    ax.legend()
    ax.grid(alpha=0.25)

    return fig


def _render_distribution_tab(base_params: Dict[str, float]) -> None:
    st.title("Fake/Real Distribution")

    st.markdown(
        """
        This tab shows how the whole item population is divided across the three AI regions:

        - Lower region: `x < Blow`
        - Middle region: `Blow < x < Bhigh`
        - Upper region: `x > Bhigh`

        The six values are joint probabilities, so together they should sum to `1`.
        """
    )

    st.markdown("## Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Ps = st.number_input(
            "Ps = P(fake)",
            min_value=0.01,
            max_value=0.99,
            value=float(base_params["Ps"]),
            step=0.01,
            format="%.3f",
            key="dist_Ps",
        )

    with col2:
        dAI = st.number_input(
            "dAI",
            min_value=0.0,
            value=float(base_params["dAI"]),
            step=0.1,
            format="%.3f",
            key="dist_dAI",
        )

    with col3:
        Blow = st.number_input(
            "Blow",
            value=float(base_params["Blow"]),
            step=0.1,
            format="%.3f",
            key="dist_Blow",
        )

    with col4:
        Bhigh = st.number_input(
            "Bhigh",
            value=float(base_params["Bhigh"]),
            step=0.1,
            format="%.3f",
            key="dist_Bhigh",
        )

    if Blow >= Bhigh:
        st.error("Blow must be smaller than Bhigh.")
        return

    result = fake_real_distribution_across_ai_regions(
        Ps=float(Ps),
        Blow=float(Blow),
        Bhigh=float(Bhigh),
        dAI=float(dAI),
    )

    rows = []

    for key, value in result.items():
        rows.append(
            {
                "Result": key,
                "Probability": round(value, 4),
                "Percentage": f"{round(value * 100, 2)}%",
            }
        )

    df = pd.DataFrame(rows)

    total_probability = sum(result.values())

    df.loc[len(df)] = {
        "Result": "TOTAL",
        "Probability": round(total_probability, 4),
        "Percentage": f"{round(total_probability * 100, 2)}%",
    }

    st.markdown("## Results")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## SDT Graph")

    fig = plot_sdt_ai_regions(
        Ps=float(Ps),
        Blow=float(Blow),
        Bhigh=float(Bhigh),
        dAI=float(dAI),
    )

    st.pyplot(fig)


inputs = _render_sidebar()

tab_labels = [cfg["label"] for cfg in TAB_CONFIGS] + ["Fake/Real Distribution"]
tabs = st.tabs(tab_labels)


for tab, cfg in zip(tabs[: len(TAB_CONFIGS)], TAB_CONFIGS):
    with tab:
        st.title("Fake News Detection Simulation")

        base_params = {**inputs["base_params"], **BASE_PAYOFFS}

        state_key = f"state_{cfg['label']}"
        sig_key = f"sig_{cfg['label']}"

        st.markdown("## Results")

        error = _validate_params(
            base_params,
            inputs["sweep_min"],
            inputs["sweep_max"],
            inputs["sweep_step"],
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

                run_result = _run_simulation(cfg["script"], payload)

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
                        denominator = (
                            tp + fn
                            if isinstance(tp, (int, float)) and isinstance(fn, (int, float))
                            else None
                        )
                        value = (tp / denominator) if denominator else None
                    else:
                        denominator = (
                            fp + tn
                            if isinstance(fp, (int, float)) and isinstance(tn, (int, float))
                            else None
                        )
                        value = (fp / denominator) if denominator else None
                else:
                    value = outcomes.get(key) if outcomes else None

                display = f"{value:.4f}" if isinstance(value, (int, float)) else "—"

                col.markdown(label)
                col.markdown(display)

            st.markdown("## Computed B* map")
            st.json(bstar_map if bstar_map is not None else {})

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

                if state["run"]["stderr"]:
                    st.code(state["run"]["stderr"], language="text")

            if state and state.get("compute_error"):
                st.error("Live metric computation failed:")
                st.code(state["compute_error"], language="text")


with tabs[-1]:
    base_params = {**inputs["base_params"], **BASE_PAYOFFS}
    _render_distribution_tab(base_params)


# To run locally:
# streamlit run simulationapp.py
