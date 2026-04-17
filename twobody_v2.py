from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

# --- Constants ---
G: float = 6.67430e-11
M_EARTH: float = 5.972e24
R_EARTH: float = 6.371e6


# --- Physics ---
def two_body(t: float, state: np.ndarray) -> np.ndarray:
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -G * M_EARTH * x / r**3
    ay = -G * M_EARTH * y / r**3
    return np.array([vx, vy, ax, ay], dtype=float)


def simulate_orbit(
    r0: float,
    v0: float,
    t_max: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_state = np.array([r0, 0.0, 0.0, v0], dtype=float)
    t_eval = np.linspace(0.0, t_max, n_steps)

    sol = solve_ivp(
        fun=two_body,
        t_span=(0.0, t_max),
        y0=initial_state,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    return sol.t, sol.y[0], sol.y[1]


# --- Plot helpers ---
def earth_trace() -> go.Scatter:
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    return go.Scatter(
        x=(R_EARTH * np.cos(theta)) / 1000.0,
        y=(R_EARTH * np.sin(theta)) / 1000.0,
        mode="lines",
        fill="toself",
        name="Earth",
        hoverinfo="skip",
    )


def compute_plot_limits(
    x_km: np.ndarray,
    y_km: np.ndarray,
    earth_radius_km: float,
    padding_fraction: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = min(float(np.min(x_km)), -earth_radius_km)
    x_max = max(float(np.max(x_km)), earth_radius_km)
    y_min = min(float(np.min(y_km)), -earth_radius_km)
    y_max = max(float(np.max(y_km)), earth_radius_km)

    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    half_span *= 1.0 + padding_fraction

    # Center on Earth for a more intuitive view
    return (-half_span, half_span), (-half_span, half_span)


def make_animation_figure(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_days: np.ndarray,
    frame_stride: int,
    frame_duration_ms: int,
) -> go.Figure:
    earth_radius_km = R_EARTH / 1000.0
    x_range, y_range = compute_plot_limits(x_km, y_km, earth_radius_km)

    # Initial traces
    fig = go.Figure(
        data=[
            earth_trace(),
            go.Scatter(
                x=[x_km[0]],
                y=[y_km[0]],
                mode="lines",
                name="Moon path",
            ),
            go.Scatter(
                x=[x_km[0]],
                y=[y_km[0]],
                mode="markers",
                name="Moon",
                marker={"size": 10},
            ),
        ]
    )

    frames: list[go.Frame] = []
    slider_steps: list[dict] = []

    frame_indices = list(range(1, len(x_km), frame_stride))
    if frame_indices[-1] != len(x_km) - 1:
        frame_indices.append(len(x_km) - 1)

    for i in frame_indices:
        frame_name = f"frame_{i}"

        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    earth_trace(),
                    go.Scatter(
                        x=x_km[: i + 1],
                        y=y_km[: i + 1],
                        mode="lines",
                        name="Moon path",
                    ),
                    go.Scatter(
                        x=[x_km[i]],
                        y=[y_km[i]],
                        mode="markers",
                        name="Moon",
                        marker={"size": 10},
                    ),
                ],
                traces=[0, 1, 2],
            )
        )

        slider_steps.append(
            {
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": frame_duration_ms, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{t_days[i]:.1f}",
                "method": "animate",
            }
        )

    fig.frames = frames

    fig.update_layout(
        title="Animated Orbit",
        xaxis_title="x (km)",
        yaxis_title="y (km)",
        showlegend=True,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis={"range": list(x_range)},
        yaxis={
            "range": list(y_range),
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "direction": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 40},
                "currentvalue": {"prefix": "Simulated time (days): "},
                "steps": slider_steps,
            }
        ],
    )

    return fig


# --- UI ---
st.set_page_config(page_title="Earth–Moon Orbit Explorer", layout="centered")

st.title("🌍🌕 Earth–Moon Orbit Explorer")
st.markdown(
    "Set an initial Earth–Moon separation and tangential speed, then click **Simulate**."
)

st.sidebar.header("Initial Conditions")

r0_km = st.sidebar.slider(
    "Initial separation (km)",
    min_value=7000,
    max_value=500000,
    value=384400,
    step=1000,
)

v0 = st.sidebar.slider(
    "Initial tangential speed (m/s)",
    min_value=0,
    max_value=2000,
    value=1022,
    step=10,
)

t_days_total = st.sidebar.slider(
    "Simulation duration (days)",
    min_value=1,
    max_value=60,
    value=30,
)

n_steps = st.sidebar.slider(
    "Number of calculation steps",
    min_value=300,
    max_value=4000,
    value=2000,
    step=100,
)

frame_stride = st.sidebar.slider(
    "Animation frame stride",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Smaller values give more frames and smoother motion.",
)

frame_duration_ms = st.sidebar.slider(
    "Frame duration (ms)",
    min_value=10,
    max_value=200,
    value=30,
    step=5,
)

if st.button("Calculate Simulation"):
    r0_m = r0_km * 1000.0
    t_max_s = t_days_total * 24.0 * 3600.0

    t_s, x_m, y_m = simulate_orbit(
        r0=r0_m,
        v0=float(v0),
        t_max=t_max_s,
        n_steps=n_steps,
    )

    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    t_days = t_s / (24.0 * 3600.0)
    r_km = np.sqrt(x_km**2 + y_km**2)

    fig = make_animation_figure(
        x_km=x_km,
        y_km=y_km,
        t_days=t_days,
        frame_stride=frame_stride,
        frame_duration_ms=frame_duration_ms,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Diagnostics")
    st.markdown(
        f"""
**Minimum distance from Earth's center:** {np.min(r_km):,.0f} km  
**Maximum distance from Earth's center:** {np.max(r_km):,.0f} km  
**Final distance from Earth's center:** {r_km[-1]:,.0f} km
"""
    )