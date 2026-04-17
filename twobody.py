from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

# --- Constants ---
G: float = 6.67430e-11          # m^3 kg^-1 s^-2
M_EARTH: float = 5.972e24       # kg
R_EARTH: float = 6.371e6        # m

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
    """
    Simulate the orbit of a test particle around Earth.

    Initial conditions:
    - position starts on +x axis
    - velocity is purely tangential in +y direction
    """
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


def make_orbit_figure(
    x_km: np.ndarray,
    y_km: np.ndarray,
    moon_index: int | None = None,
    title: str = "Orbit Trajectory",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(earth_trace())

    fig.add_trace(
        go.Scatter(
            x=x_km,
            y=y_km,
            mode="lines",
            name="Moon path",
        )
    )

    if moon_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[x_km[moon_index]],
                y=[y_km[moon_index]],
                mode="markers",
                name="Moon",
                marker={"size": 10},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="x (km)",
        yaxis_title="y (km)",
        yaxis_scaleanchor="x",
        showlegend=True,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
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

t_days = st.sidebar.slider(
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
    "Animation smoothness",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Smaller values give more animation frames but may run more slowly.",
)

frame_delay = st.sidebar.slider(
    "Animation delay per frame (seconds)",
    min_value=0.00,
    max_value=0.20,
    value=0.02,
    step=0.01,
)

# Convert units
r0_m = r0_km * 1000.0
t_max_s = t_days * 24.0 * 3600.0

# Preview initial configuration
st.subheader("Initial setup")
preview_fig = make_orbit_figure(
    x_km=np.array([r0_km]),
    y_km=np.array([0.0]),
    moon_index=0,
    title="Initial Position",
)
st.plotly_chart(preview_fig, use_container_width=True)

simulate_button = st.button("Simulate")

if simulate_button:
    t, x, y = simulate_orbit(
        r0=r0_m,
        v0=float(v0),
        t_max=t_max_s,
        n_steps=n_steps,
    )

    x_km = x / 1000.0
    y_km = y / 1000.0
    r_km = np.sqrt(x_km**2 + y_km**2)

    animation_placeholder = st.empty()
    status_placeholder = st.empty()

    st.subheader("Animation")

    for i in range(1, len(x_km), frame_stride):
        animated_fig = make_orbit_figure(
            x_km=x_km[: i + 1],
            y_km=y_km[: i + 1],
            moon_index=i,
            title="Animated Orbit",
        )

        animation_placeholder.plotly_chart(
            animated_fig,
            use_container_width=True,
            key=f"anim_frame_{i}",
        )

        elapsed_days = t[i] / (24.0 * 3600.0)
        status_placeholder.markdown(f"**Simulated time:** {elapsed_days:.2f} days")

        time.sleep(frame_delay)

    # Ensure final frame is shown
    final_anim_fig = make_orbit_figure(
        x_km=x_km,
        y_km=y_km,
        moon_index=len(x_km) - 1,
        title="Animated Orbit",
    )
    animation_placeholder.plotly_chart(
        final_anim_fig,
        use_container_width=True,
        key="anim_final",
    )

    st.subheader("Full trajectory")
    full_fig = make_orbit_figure(
        x_km=x_km,
        y_km=y_km,
        moon_index=None,
        title="Orbit Trajectory",
    )
    st.plotly_chart(full_fig, use_container_width=True)

    st.subheader("Diagnostics")
    st.markdown(
        f"""
**Minimum distance from Earth's center:** {np.min(r_km):,.0f} km  
**Maximum distance from Earth's center:** {np.max(r_km):,.0f} km  
**Final distance from Earth's center:** {r_km[-1]:,.0f} km
"""
    )
