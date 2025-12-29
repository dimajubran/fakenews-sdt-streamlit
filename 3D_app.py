from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sim_core import compute_bstar_map, compute_outcomes


st.set_page_config(page_title="SDT Simulation Dashboard (3D)", layout="wide")

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


PARAM_OPTIONS = ["Ps", "dH", "dAI", "Blow", "Bhigh"]


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


def _default_range(param_name: str, current_value: float) -> Tuple[float, float, float]:
    default_min = current_value - 1.0
    default_max = current_value + 1.0
    default_step = 0.1
    if param_name == "Ps":
        default_min = 0.05
        default_max = 0.95
        default_step = 0.01
    return default_min, default_max, default_step


def _axis_controls(axis_label: str, base_params: Dict[str, float], default_index: int) -> Dict[str, float]:
    st.sidebar.subheader(axis_label)
    param_name = st.sidebar.selectbox(
        f"{axis_label} parameter",
        PARAM_OPTIONS,
        index=default_index,
        key=f"{axis_label}_param",
    )

    current_value = float(base_params[param_name])
    default_min, default_max, default_step = _default_range(param_name, current_value)

    sweep_min = st.sidebar.number_input(
        f"{axis_label} min",
        value=default_min,
        step=default_step,
        format="%.2f",
        key=f"{axis_label}_min",
    )
    sweep_max = st.sidebar.number_input(
        f"{axis_label} max",
        value=default_max,
        step=default_step,
        format="%.2f",
        key=f"{axis_label}_max",
    )
    sweep_step = st.sidebar.number_input(
        f"{axis_label} step",
        min_value=1e-6,
        value=default_step,
        step=default_step,
        format="%.2f",
        key=f"{axis_label}_step",
    )

    return {
        "param_name": param_name,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "sweep_step": sweep_step,
    }


@st.cache_data(show_spinner=False)
def _cached_surface(
    base_params: Dict[str, float],
    param_x: str,
    x_min: float,
    x_max: float,
    x_step: float,
    param_y: str,
    y_min: float,
    y_max: float,
    y_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.arange(x_min, x_max + 0.5 * x_step, x_step)
    y_values = np.arange(y_min, y_max + 0.5 * y_step, y_step)
    z_values = np.zeros((len(y_values), len(x_values)))

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            params = dict(base_params)
            params[param_x] = float(x_val)
            params[param_y] = float(y_val)
            out = compute_outcomes(params)
            z_values[i, j] = out["Score"]

    return x_values, y_values, z_values


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

st.sidebar.header("Parameter Surface")
axis_x = _axis_controls("X-axis", base_params, default_index=0)
axis_y = _axis_controls("Y-axis", base_params, default_index=1)

if axis_x["param_name"] == axis_y["param_name"]:
    st.error("Choose two different parameters for the surface plot.")
    st.stop()

if axis_x["sweep_min"] >= axis_x["sweep_max"] or axis_y["sweep_min"] >= axis_y["sweep_max"]:
    st.error("Axis min must be less than axis max.")
    st.stop()

if axis_x["sweep_step"] <= 0 or axis_y["sweep_step"] <= 0:
    st.error("Axis step sizes must be greater than 0.")
    st.stop()

if axis_x["param_name"] == "Ps" and not (0 < axis_x["sweep_min"] < axis_x["sweep_max"] < 1):
    st.error("Ps range must be within (0, 1).")
    st.stop()

if axis_y["param_name"] == "Ps" and not (0 < axis_y["sweep_min"] < axis_y["sweep_max"] < 1):
    st.error("Ps range must be within (0, 1).")
    st.stop()

x_values, y_values, z_values = _cached_surface(
    base_params_with_bstar,
    axis_x["param_name"],
    axis_x["sweep_min"],
    axis_x["sweep_max"],
    axis_x["sweep_step"],
    axis_y["param_name"],
    axis_y["sweep_min"],
    axis_y["sweep_max"],
    axis_y["sweep_step"],
)

st.subheader("Score Surface")
plot_col, _ = st.columns([0.7, 0.3])

fig = go.Figure(
    data=[
        go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale="Viridis",
        )
    ]
)
fig.update_layout(
    title=f"Score surface: {axis_x['param_name']} vs {axis_y['param_name']} (all others fixed at base case)",
    scene=dict(
        xaxis_title=axis_x["param_name"],
        yaxis_title=axis_y["param_name"],
        zaxis_title="Score",
    ),
    margin=dict(l=10, r=10, b=10, t=50),
    height=520,
)

plot_col.plotly_chart(fig, use_container_width=True)
