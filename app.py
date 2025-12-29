from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sim_core import compute_bstar_map, compute_outcomes, sweep_param


st.set_page_config(page_title="SDT Simulation Dashboard", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 14px; }
    h1 { font-size: 28px; }
    h2 { font-size: 20px; }
    h3 { font-size: 16px; }
    .stMetric label { font-size: 12px; }
    .stMetric div { font-size: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _build_base_params() -> Dict[str, float]:
    st.sidebar.header("Base Case Parameters")
    Ps = st.sidebar.number_input("Ps", min_value=0.001, max_value=0.999, value=0.2, step=0.01, format="%.2f")
    dH = st.sidebar.number_input("dH", min_value=0.01, value=2.5, step=0.1, format="%.2f")
    dAI = st.sidebar.number_input("dAI", min_value=0.0, value=2.5, step=0.1, format="%.2f")
    Blow = st.sidebar.number_input("Blow", value=-2.0, step=0.1, format="%.2f")
    Bhigh = st.sidebar.number_input("Bhigh", value=2.0, step=0.1, format="%.2f")
    V_TP = st.sidebar.number_input("V_TP", value=100.0, step=10.0, format="%.2f")
    V_TN = st.sidebar.number_input("V_TN", value=100.0, step=10.0, format="%.2f")
    V_FP = st.sidebar.number_input("V_FP", value=-100.0, step=10.0, format="%.2f")
    V_FN = st.sidebar.number_input("V_FN", value=-100.0, step=10.0, format="%.2f")

    return {
        "Ps": Ps,
        "dH": dH,
        "dAI": dAI,
        "Blow": Blow,
        "Bhigh": Bhigh,
        "V_TP": V_TP,
        "V_TN": V_TN,
        "V_FP": V_FP,
        "V_FN": V_FN,
    }


def _build_sweep_controls(base_params: Dict[str, float]) -> Dict[str, float]:
    st.sidebar.header("Parameter Sweep")
    param_name = st.sidebar.selectbox(
        "Sweep parameter",
        ["Ps", "dH", "dAI", "Blow", "Bhigh", "V_TP", "V_TN", "V_FP", "V_FN"],
        index=0,
    )

    current_value = float(base_params[param_name])
    default_min = current_value - 1.0
    default_max = current_value + 1.0
    default_step = 0.1
    if param_name == "Ps":
        default_min = 0.05
        default_max = 0.95
        default_step = 0.01

    sweep_min = st.sidebar.number_input(
        "Sweep min",
        value=default_min,
        step=default_step,
        format="%.2f",
    )
    sweep_max = st.sidebar.number_input(
        "Sweep max",
        value=default_max,
        step=default_step,
        format="%.2f",
    )
    sweep_step = st.sidebar.number_input(
        "Sweep step",
        min_value=1e-6,
        value=default_step,
        step=default_step,
        format="%.2f",
    )

    return {
        "param_name": param_name,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "sweep_step": sweep_step,
    }


@st.cache_data(show_spinner=False)
def _cached_sweep(
    base_params: Dict[str, float],
    param_name: str,
    sweep_min: float,
    sweep_max: float,
    sweep_step: float,
) -> pd.DataFrame:
    values = np.arange(sweep_min, sweep_max + 0.5 * sweep_step, sweep_step)
    return sweep_param(base_params, param_name, values)


st.title("Fake News Detection Simulation")

base_params = _build_base_params()

if base_params["Blow"] >= base_params["Bhigh"]:
    st.error("Require Blow < Bhigh for a valid base case.")
    st.stop()

bstar_map = compute_bstar_map(base_params)
base_params_with_bstar = dict(base_params)
base_params_with_bstar["Bstar_map"] = bstar_map

base_outcomes = compute_outcomes(base_params_with_bstar)

st.subheader("Results")

metric_cols = st.columns(5)
metric_cols[0].metric("Score", f"{base_outcomes['Score']:.4f}")
metric_cols[1].metric("TP", f"{base_outcomes['TP']:.4f}")
metric_cols[2].metric("TN", f"{base_outcomes['TN']:.4f}")
metric_cols[3].metric("FP", f"{base_outcomes['FP']:.4f}")
metric_cols[4].metric("FN", f"{base_outcomes['FN']:.4f}")

st.subheader("Computed B* map")
st.json(bstar_map)

sweep_controls = _build_sweep_controls(base_params)

if sweep_controls["sweep_min"] >= sweep_controls["sweep_max"]:
    st.error("Sweep min must be less than sweep max.")
    st.stop()

sweep_df = _cached_sweep(
    base_params_with_bstar,
    sweep_controls["param_name"],
    sweep_controls["sweep_min"],
    sweep_controls["sweep_max"],
    sweep_controls["sweep_step"],
)

st.subheader("Score vs Parameter Sweep")
plot_col, _ = st.columns([0.55, 0.45])
fig, ax = plt.subplots(figsize=(3.4, 2.0))
ax.plot(sweep_df[sweep_controls["param_name"]], sweep_df["Score"], marker="o")
ax.set_xlabel(sweep_controls["param_name"])
ax.set_ylabel("Score")
ax.set_title(f"Score vs {sweep_controls['param_name']}")
ax.grid(True, alpha=0.3)

plot_col.pyplot(fig, clear_figure=True)
