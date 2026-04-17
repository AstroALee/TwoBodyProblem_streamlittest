from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

# --- Constants ---
G: float = 6.67430e-11
M_EARTH: float = 5.972e24
M_MOON: float = 7.34767309e22
R_EARTH: float = 6.371e6
R_MOON: float = 1.7374e6

T_MAX_DAYS: float = 90.0
T_MAX_S: float = T_MAX_DAYS * 24.0 * 3600.0
N_STEPS: int = 2000
FRAME_STRIDE: int = 10
FRAME_DURATION_MS: int = 60


# --- Physics ---
def two_body_fixed_earth(t: float, state: np.ndarray) -> np.ndarray:
    x, y, vx, vy = state
    r = np.hypot(x, y)
    ax = -G * M_EARTH * x / r**3
    ay = -G * M_EARTH * y / r**3
    return np.array([vx, vy, ax, ay], dtype=float)


def orbital_elements_from_initial_conditions(r0: float, v0: float):
    mu = G * M_EARTH
    epsilon = 0.5 * v0**2 - mu / r0
    h = r0 * v0
    e = np.sqrt(1.0 + 2.0 * epsilon * h**2 / mu**2)
    a = np.inf if np.isclose(epsilon, 0.0) else -mu / (2.0 * epsilon)
    return epsilon, a, e


def simulate_orbit(r0: float, v0: float):
    initial_state = np.array([r0, 0.0, 0.0, v0], dtype=float)
    t_eval = np.linspace(0.0, T_MAX_S, N_STEPS)

    sol = solve_ivp(
        fun=two_body_fixed_earth,
        t_span=(0.0, T_MAX_S),
        y0=initial_state,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    return sol.t, sol.y[0], sol.y[1]


def collision_index(r_m: np.ndarray, point_particles: bool):
    if point_particles:
        return None

    collision_radius = R_EARTH + R_MOON
    hits = np.where(r_m <= collision_radius)[0]
    return int(hits[0]) if len(hits) else None


# --- Plot helpers ---
def earth_trace():
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    return go.Scatter(
        x=(R_EARTH * np.cos(theta)) / 1000.0,
        y=(R_EARTH * np.sin(theta)) / 1000.0,
        mode="lines",
        fill="toself",
        name="Earth",
        hoverinfo="skip",
    )


def compute_plot_limits(x_km: np.ndarray, y_km: np.ndarray):
    max_extent = max(
        float(np.max(np.abs(x_km))),
        float(np.max(np.abs(y_km))),
        R_EARTH / 1000.0,
    )
    span = max_extent * 1.08
    return (-span, span), (-span, span)


def make_animation_figure(x_km, y_km, t_days):
    x_range, y_range = compute_plot_limits(x_km, y_km)

    fig = go.Figure(
        data=[
            earth_trace(),
            go.Scatter(x=[x_km[0]], y=[y_km[0]], mode="lines", name="Path"),
            go.Scatter(x=[x_km[0]], y=[y_km[0]], mode="markers", name="Moon"),
        ]
    )

    frames = []
    slider_steps = []

    indices = list(range(1, len(x_km), FRAME_STRIDE))
    if indices[-1] != len(x_km) - 1:
        indices.append(len(x_km) - 1)

    for i in indices:
        name = f"f{i}"

        frames.append(
            go.Frame(
                name=name,
                data=[
                    earth_trace(),
                    go.Scatter(x=x_km[: i + 1], y=y_km[: i + 1], mode="lines"),
                    go.Scatter(x=[x_km[i]], y=[y_km[i]], mode="markers"),
                ],
            )
        )

        slider_steps.append(
            {
                "args": [[name], {"frame": {"duration": FRAME_DURATION_MS, "redraw": False}}],
                "label": f"{t_days[i]:.1f}",
                "method": "animate",
            }
        )

    fig.frames = frames

    fig.update_layout(
        title="Animated Orbit",
        margin={"l": 20, "r": 20, "t": 95, "b": 20},
        updatemenus=[{
            "type": "buttons",
            "x": 0.05,
            "y": 1.05,
            "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": FRAME_DURATION_MS}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "steps": slider_steps,
            "currentvalue": {"prefix": "Time (days): "},
        }],
    )

    fig.update_xaxes(
        range=list(x_range),
        showgrid=True,
        griddash="dash",
        gridcolor="rgba(120,120,120,0.3)",
        tickfont=dict(size=14),
    )

    fig.update_yaxes(
        range=list(y_range),
        scaleanchor="x",
        showgrid=True,
        griddash="dash",
        gridcolor="rgba(120,120,120,0.3)",
        tickfont=dict(size=14),
    )

    return fig


# --- App ---
st.title("🌍🌕 Earth–Moon Orbit Explorer")

r0_km = st.slider("Initial separation (km)", 7000, 500000, 384400, step=1000)
v0 = st.slider("Initial tangential speed (m/s)", 0, 2000, 1022, step=10)

point_particles = st.checkbox("Treat as point particles", value=False)

if st.button("Determine Trajectory"):
    r0 = r0_km * 1000.0

    epsilon, a, e = orbital_elements_from_initial_conditions(r0, v0)

    t, x, y = simulate_orbit(r0, v0)
    r = np.hypot(x, y)

    hit = collision_index(r, point_particles)

    if hit is not None:
        x, y, t, r = x[:hit], y[:hit], t[:hit], r[:hit]

    x_km = x / 1000.0
    y_km = y / 1000.0
    t_days = t / (24 * 3600)

    fig = make_animation_figure(x_km, y_km, t_days)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Results")

    if hit is not None:
        st.write("**Outcome:** Collision")
    elif epsilon >= 0:
        st.write("**Outcome:** Unbound orbit")
    else:
        st.write(f"**Semi-major axis:** {a/1000:,.0f} km")
        st.write(f"**Eccentricity:** {e:.4f}")