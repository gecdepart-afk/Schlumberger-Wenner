# ==============================
# 2D DC Forward Modelling (SimPEG)
# Streamlit app — dipole–dipole forward modelling + 2 pseudosections
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from discretize import TensorMesh
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

# ---------------------------
# 1) PAGE SETUP
# ---------------------------

st.set_page_config(page_title="2D DC Forward (SimPEG)", page_icon="🌍", layout="wide")
st.title("2D DC Resistivity – dipole–dipole forward modelling (SimPEG)")
st.markdown(
    "Simple 2D model: background + rectangular anomaly, dipole–dipole line, "
    "forward modelled with SimPEG and plotted as both a **discrete** and an "
    "**interpolated** pseudosection."
)

# ---------------------------
# 2) SIDEBAR – MODEL & SURVEY
# ---------------------------

with st.sidebar:
    st.header("Model parameters")

    rho_bg = st.number_input("Background resistivity ρ_bg (Ω·m)", min_value=1.0, value=100.0, step=10.0)
    rho_block = st.number_input("Block resistivity ρ_block (Ω·m)", min_value=1.0, value=10.0, step=5.0)

    st.markdown("**Block position (x in m, depth z > 0 downward)**")
    block_xmin = st.number_input("Block x_min (m)", value=-20.0, step=5.0)
    block_xmax = st.number_input("Block x_max (m)", value=20.0, step=5.0)
    block_zmin = st.number_input("Block top depth (m)", value=5.0, step=1.0)
    block_zmax = st.number_input("Block bottom depth (m)", value=25.0, step=1.0)

    st.divider()
    st.header("Survey geometry")

    line_length = st.number_input("Line length (m)", min_value=20.0, value=100.0, step=10.0)
    n_electrodes = st.slider("Number of electrodes", min_value=16, max_value=72, value=32, step=4)
    n_max = st.slider("Maximum n-spacing (dipole–dipole)", 1, 8, 4)

    st.caption(
        "Dipole–dipole along a straight line at z = 0. "
        "For each electrode, n = 1..n_max are generated, giving the usual "
        "triangular pseudosection."
    )

# ---------------------------
# 3) BUILD 2D MESH & MODEL
# ---------------------------

# horizontal electrodes from -L/2 to +L/2
x_electrodes = np.linspace(-line_length / 2.0, line_length / 2.0, n_electrodes)

# 2D mesh: x (horizontal), z (vertical, depth > 0 downward in physical sense)
domain_width = line_length * 1.5
domain_depth = line_length

nx = 80
nz = 40
hx = np.ones(nx) * (domain_width / nx)
hz = np.ones(nz) * (domain_depth / nz)

# origin at left, surface z = 0; cells extend downward (negative mesh coords)
mesh = TensorMesh([hx, hz], x0=(-domain_width / 2.0, -domain_depth))

# background model
rho_model = rho_bg * np.ones(mesh.nC)

xc, zc = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]  # zc is negative (downwards)

# convert block depths (z > 0) to mesh coordinates (negative)
z_block_top = -block_zmin
z_block_bottom = -block_zmax

in_block = (
    (xc >= block_xmin)
    & (xc <= block_xmax)
    & (zc <= z_block_top)
    & (zc >= z_block_bottom)
)

rho_model[in_block] = rho_block

rho_map = maps.IdentityMap(nP=mesh.nC)

# ---------------------------
# 4) BUILD DIPOLE–DIPOLE SURVEY (full pseudosection)
# ---------------------------

src_list = []
midpoints = []
pseudo_depths = []

# electrode spacing (assumed uniform)
dx = x_electrodes[1] - x_electrodes[0]  # this is "a" (one electrode spacing)

for iA in range(n_electrodes):
    for n in range(1, n_max + 1):
        iB = iA + 1
        iM = iA + 1 + n
        iN = iA + 2 + n

        if iN >= n_electrodes:
            break  # no more valid configurations for this iA

        # 2D coordinates (x, z) for a 2D simulation
        A = np.r_[x_electrodes[iA], 0.0]
        B = np.r_[x_electrodes[iB], 0.0]
        M = np.r_[x_electrodes[iM], 0.0]
        N = np.r_[x_electrodes[iN], 0.0]

        rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
        src = dc.sources.Dipole([rx], A, B)
        src_list.append(src)

        # standard dipole–dipole pseudosection coordinates
        midpoints.append(0.5 * (M[0] + N[0]))
        pseudo_depths.append(0.5 * n * dx)  # pseudo-depth ≈ n·a / 2

midpoints = np.array(midpoints)
pseudo_depths = np.array(pseudo_depths)

survey = dc.Survey(src_list)
survey.set_geometric_factor(space_type="halfspace")

# ---------------------------
# 5) SIMULATION & FORWARD
# ---------------------------

try:
    # NOTE: if your SimPEG version does not have Simulation2DNodal,
    # change to dc.Simulation2DCellCentered or dc.Problem2D_CC accordingly.
    sim = dc.Simulation2DNodal(
        mesh=mesh,
        survey=survey,
        rhoMap=rho_map,
    )

    data = sim.dpred(rho_model)  # apparent resistivities
    ok = True
except Exception as e:
    ok = False
    st.error(f"Forward modelling failed: {e}")

# ---------------------------
# 6) DISPLAY: MODEL + TWO PSEUDOSECTIONS
# ---------------------------

col1, col2 = st.columns([1.1, 1.9])

# --- LEFT: 2D resistivity model ---
with col1:
    st.subheader("Resistivity model (2D)")

    # convert mesh coords (negative) to physical depth > 0
    z_depth = -zc

    fig_m, ax_m = plt.subplots(figsize=(4, 5))
    m_img = ax_m.tripcolor(
        xc, z_depth, np.log10(rho_model),
        shading="gouraud"
    )
    ax_m.invert_yaxis()  # depth increases downward
    ax_m.set_xlabel("x (m)")
    ax_m.set_ylabel("Depth (m)")
    ax_m.set_title("log10(ρ) model")
    fig_m.colorbar(m_img, ax=ax_m, label="log10(ρ / Ω·m)")
    st.pyplot(fig_m, clear_figure=True)

    st.caption(
        "Background + rectangular anomaly. Depth is positive downward. "
        "You can change ρ_bg, ρ_block and block size/position in the sidebar."
    )

# --- RIGHT: discrete + interpolated pseudosections ---
with col2:
    st.subheader("Dipole–dipole apparent resistivity pseudosections")

    if ok:
        fig_d, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # 1) Discrete pseudosection (scatter)
        sc1 = ax_s.scatter(
            midpoints,
            pseudo_depths,
            c=data,
            cmap="viridis",
            s=40,
            edgecolors="k"
        )
        ax_s.invert_yaxis()
        ax_s.set_xlabel("Midpoint (m)")
        ax_s.set_ylabel("Pseudo-depth (m)")
        ax_s.set_title("Discrete pseudosection")
        fig_d.colorbar(sc1, ax=ax_s, label="ρₐ (Ω·m)")

        # 2) Interpolated pseudosection (tricontourf)
        levels = 20
        cf = ax_i.tricontourf(
            midpoints,
            pseudo_depths,
            data,
            levels=levels,
            cmap="viridis"
        )
        ax_i.invert_yaxis()
        ax_i.set_xlabel("Midpoint (m)")
        ax_i.set_title("Interpolated pseudosection")
        fig_d.colorbar(cf, ax=ax_i, label="ρₐ (Ω·m)")

        for ax in (ax_s, ax_i):
            ax.grid(True, linestyle=":", alpha=0.5)

        fig_d.tight_layout()
        st.pyplot(fig_d, clear_figure=True)

        st.caption(
            "Left: individual measurements (standard dipole–dipole layout). "
            "Right: interpolated pseudosection (closer to what RES2DINV displays). "
            "Vertical axis is a pseudo-depth n·a/2, not a true inversion depth."
        )

# ---------------------------
# 7) NOTES
# ---------------------------

st.divider()
st.caption(
    "The app uses a 2D TensorMesh and a dipole–dipole line with all n-spacings "
    "from 1 to n_max, generating the classical triangular pseudosection. "
    "Apparent resistivities are computed by SimPEG using the half-space "
    "geometric factor."
)
