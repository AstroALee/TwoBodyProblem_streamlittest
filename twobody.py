from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

# --- Constants ---
G: float = 6.67430e-11          # m^3 kg^-1 s^-2
M_EARTH: float = 5.972e24       # kg
R_EARTH: float = 6.371e6        # m

# --- Physics ---
def two_body(t: float, y: np.ndarray) -> np.ndarray:
    x, y_pos, vx, vy = y
    r = np.sqrt(x**2 + y_pos**2)
    ax = -G * M_EARTH * x / r**3
    ay = -G * M_EARTH * y_pos / r**3
    return np.array([vx, vy, ax, ay])

def simulate(
    r0: float,
    v0: float,
    t_max: float,
    n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    # Initial state: start on x-axis, velocity purely tangential (y-direction)
    y0 = np.array([r0, 0.0, 0.0, v0])

    t_eval = np.linspace(0, t_max, n_steps)

    sol = solve_ivp(
        two_body,
        (0, t_max),
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    return sol.y[0], sol.y[1]  # x(t), y(t)

# --- UI ---
st.title("🌍🌕 Earth–Moon Orbit Explorer")

st.markdown("Adjust the initial distance and speed to explore different orbits.")

# Sidebar controls
st.sidebar.header("Initial Conditions")

r0_km = st.sidebar.slider(
    "Initial separation (km)",
    min_value=7000,
    max_value=500000,
    value=384400,
    step=1000
)

v0 = st.sidebar.slider(
    "Initial tangential speed (m/s)",
    min_value=0,
    max_value=2000,
    value=1022,
    step=10
)

t_days = st.sidebar.slider(
    "Simulation duration (days)",
    min_value=1,
    max_value=60,
    value=30
)

# Convert units
r0 = r0_km * 1000
t_max = t_days * 24 * 3600

# Run simulation
x, y = simulate(r0, v0, t_max, n_steps=2000)

# --- Plot ---
fig = go.Figure()

# Orbit path
fig.add_trace(go.Scatter(
    x=x / 1000,
    y=y / 1000,
    mode="lines",
    name="Moon path"
))

# Earth
theta = np.linspace(0, 2*np.pi, 200)
fig.add_trace(go.Scatter(
    x=(R_EARTH * np.cos(theta)) / 1000,
    y=(R_EARTH * np.sin(theta)) / 1000,
    fill="toself",
    name="Earth"
))

fig.update_layout(
    xaxis_title="x (km)",
    yaxis_title="y (km)",
    yaxis_scaleanchor="x",
    title="Orbit Trajectory",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# --- Diagnostics ---
r = np.sqrt(x**2 + y**2)
r_min = np.min(r) / 1000
r_max = np.max(r) / 1000

st.markdown(f"""
**Perigee:** {r_min:,.0f} km  
**Apogee:** {r_max:,.0f} km  
""")
