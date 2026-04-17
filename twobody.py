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


def orbital_elements_from_initial_conditions(r0: float, v0: float) -> tuple[float, float, float]:
    """
    Return (specific_energy, semi_major_axis, eccentricity)
    for the initial state assuming Earth fixed at the origin and
    a purely tangential initial velocity.
    """
    mu = G * M_EARTH

    # Specific orbital energy
    epsilon = 0.5 * v0**2 - mu / r0

    # Specific angular momentum magnitude
    h = r0 * v0

    # Eccentricity
    e = np.sqrt(1.0 + 2.0 * epsilon * h**2 / mu**2)

    # Semi-major axis
    if np.isclose(epsilon, 0.0):
        a = np.inf
    else:
        a = -mu / (2.0 * epsilon)

    return epsilon, a, e


def simulate_orbit(
    r0: float,
    v0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def collision_index(r_m: np.ndarray, treat_as_point_particles: bool) -> int | None:
    """
    Return the index of the first collision sample, or None if no collision.
    If a collision occurs, later we will stop on the last pre-collision frame.
    """
    if treat_as_point_particles:
        collision_radius = 0.0
    else:
        collision_radius = R_EARTH + R_MOON

    if collision_radius <= 0.0:
        return None

    hit_indices = np.where(r_m <= collision_radius)[0]
    if len(hit_indices) == 0:
        return None
    return int(hit_indices[0])


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
    padding_fraction: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Keep Earth centered and use a square view.
    """
    max_extent = max(
        float(np.max(np.abs(x_km))),
        float(np.max(np.abs(y_km))),
        R_EARTH / 1000.0,
    )
    half_span = max_extent * (1.0 + padding_fraction)
    return (-half_span, half_span), (-half_span, half_span)


def make_animation_figure(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_days: np.ndarray,
) -> go.Figure:
    x_range, y_range = compute_plot_limits(x_km, y_km)

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

    frame_indices = list(range(1, len(x_km), FRAME_STRIDE))
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
                        "frame": {"duration": FRAME_DURATION_MS, "redraw": False},
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
        margin={"l": 20, "r": 20, "t": 95, "b": 20},
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
                "y": 1.05,
                "yanchor": "top",
                "direction": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": FRAME_DURATION_MS, "redraw": False},
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

    fig.update_xaxes(
        showgrid=True,
        griddash="dash",
        gridwidth=1,
        gridcolor="rgba(120, 120, 120, 0.3)",
        tickfont={"size": 14},
    )
    fig.update_yaxes(
        showgrid=True,
        griddash="dash",
        gridwidth=1,
        gridcolor="rgba(120, 120, 120, 0.3)",
        tickfont={"size": 14},
    )

    return fig


# --- UI sync helpers ---
def sync_from_slider(key_slider: str, key_number: str) -> None:
    st.session_state[key_number] = st.session_state[key_slider]


def sync_from_number(key_slider: str, key_number: str) -> None:
    st.session_state[key_slider] = st.session_state[key_number]


# --- App ---
st.set_page_config(page_title="Earth–Moon Orbit Explorer", layout="centered")

st.title("🌍🌕 Earth–Moon Orbit Explorer")
st.markdown(
    "Choose an initial Earth–Moon separation and tangential speed, then click "
    "**Determine Trajectory**. The simulation duration is fixed at **90 days**."
)

# Initialize session state
if "r0_slider" not in st.session_state:
    st.session_state["r0_slider"] = 384400
if "r0_number" not in st.session_state:
    st.session_state["r0_number"] = 384400

if "v0_slider" not in st.session_state:
    st.session_state["v0_slider"] = 1022
if "v0_number" not in st.session_state:
    st.session_state["v0_number"] = 1022

st.subheader("Initial Conditions")

col1, col2 = st.columns([2, 1])
with col1:
    st.slider(
        "Initial separation (km)",
        min_value=7000,
        max_value=500000,
        step=1000,
        key="r0_slider",
        on_change=sync_from_slider,
        args=("r0_slider", "r0_number"),
    )
with col2:
    st.number_input(
        "Type value (km)",
        min_value=7000,
        max_value=500000,
        step=1000,
        key="r0_number",
        on_change=sync_from_number,
        args=("r0_slider", "r0_number"),
    )

col3, col4 = st.columns([2, 1])
with col3:
    st.slider(
        "Initial tangential speed (m/s)",
        min_value=0,
        max_value=2000,
        step=10,
        key="v0_slider",
        on_change=sync_from_slider,
        args=("v0_slider", "v0_number"),
    )
with col4:
    st.number_input(
        "Type value (m/s)",
        min_value=0,
        max_value=2000,
        step=10,
        key="v0_number",
        on_change=sync_from_number,
        args=("v0_slider", "v0_number"),
    )

treat_as_point_particles = st.checkbox(
    "Treat Earth and Moon as point particles",
    value=False,
    help="If unchecked, a collision is detected when the two bodies would physically touch.",
)

if st.button("Determine Trajectory"):
    r0_m = float(st.session_state["r0_number"]) * 1000.0
    v0 = float(st.session_state["v0_number"])

    epsilon, a_m, e = orbital_elements_from_initial_conditions(r0=r0_m, v0=v0)

    t_s, x_m, y_m = simulate_orbit(r0=r0_m, v0=v0)
    r_m = np.hypot(x_m, y_m)

    hit_index = collision_index(
        r_m=r_m,
        treat_as_point_particles=treat_as_point_particles,
    )

    if hit_index is not None:
        # Stop at the last pre-collision frame
        final_index = max(hit_index - 1, 0)
        x_m = x_m[: final_index + 1]
        y_m = y_m[: final_index + 1]
        t_s = t_s[: final_index + 1]
        r_m = r_m[: final_index + 1]

    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    t_days = t_s / (24.0 * 3600.0)

    fig = make_animation_figure(
        x_km=x_km,
        y_km=y_km,
        t_days=t_days,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Results")

    if hit_index is not None:
        st.markdown("**Outcome:** Collision before 90 days.")
        st.markdown(
            f"**Last pre-collision simulated time:** {t_days[-1]:.2f} days"
        )
    elif epsilon >= 0.0:
        st.markdown("**Outcome:** Unbound orbit.")
        st.markdown(
            f"**Final simulated time:** {t_days[-1]:.2f} days"
        )
    else:
        st.markdown("**Outcome:** Bound orbit.")
        st.markdown(f"**Semi-major axis:** {a_m / 1000.0:,.0f} km")
        st.markdown(f"**Eccentricity:** {e:.4f}")

        period = 2.0 * np.pi * np.sqrt(a_m**3 / (G * M_EARTH))
        st.markdown(f"**Orbital period:** {period / (24.0 * 3600.0):.2f} days")

    st.markdown(f"**Minimum distance from Earth's center:** {np.min(r_m) / 1000.0:,.0f} km")
    st.markdown(f"**Maximum distance from Earth's center:** {np.max(r_m) / 1000.0:,.0f} km")
    st.markdown(f"**Final distance from Earth's center:** {r_m[-1] / 1000.0:,.0f} km")