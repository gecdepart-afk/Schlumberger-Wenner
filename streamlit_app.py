# ==============================
# 2D DC Forward Modelling (SimPEG)
# Streamlit app — simple dipole–dipole pseudosection
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
    "forward modelled with SimPEG and plotted as a pseudosection."
)

# ---------------------------
# 2) SIDEBAR – MODEL & SURVEY
# ---------------------------

with st.sidebar:
    st.header("Model parameters")

    rho_bg = st.number_input("Background resistivity ρ_bg (Ω·m)", min_value=1.0, value=100.0, step=10.0)
    rho_block = st.number_input("Block resistivity ρ_block (Ω·m)", min_value=1.0, value=10.0, step=5.0)

    st.markdown("**Block position (x in m, z positive downward)**")
    block_xmin = st.number_input("Block x_min (m)", value=-20.0, step=5.0)
    block_xmax = st.number_input("Block x_max (m)", value=20.0, step=5.0)
    block_zmin = st.number_input("Block top depth (m)", value=5.0, step=1.0)
    block_zmax = st.number_input("Block bottom depth (m)", value=25.0, step=1.0)

    st.divider()
    st.header("Survey geometry")

    line_length = st.number_input("Line length (m)", min_value=20.0, value=100.0, step=10.0)
    n_electrodes = st.slider("Number of electrodes", min_value=16, max_value=72, value=32, step=4)
    a_factor = st.slider("Dipole separation a (in electrode spacings)", 1, 3, 1)
    n_spacing = st.slider("n-spacing (dipole–dipole)", 1, 6, 3)

    st.caption(
        "Dipole–dipole along a straight line at z=0. "
        "A–B is one dipole, M–N another; n controls separation between dipoles."
    )

# ---------------------------
# 3) BUILD 2D MESH & MODEL
# ---------------------------

# horizontal electrodes from -L/2 to +L/2
x_electrodes = np.linspace(-line_length / 2.0, line_length / 2.0, n_electrodes)
z_electrodes = np.zeros_like(x_electrodes)  # conceptually z=0

# 2D mesh: x (horizontal), z (vertical, positive down in physical sense, negative in mesh coords)
domain_width = line_length * 1.5
domain_depth = line_length

nx = 80
nz = 40
hx = np.ones(nx) * (domain_width / nx)
hz = np.ones(nz) * (domain_depth / nz)

# origin at left, top = z=0 → in mesh coords that is z=0; cells go down negative
mesh = TensorMesh([hx, hz], x0=(-domain_width / 2.0, -domain_depth))

# build resistivity model: background + rectangular block
rho_model = rho_bg * np.ones(mesh.nC)

xc, zc = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1]  # zc is negative downward

# convert block z (positive down) to mesh coords (negative)
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
# 4) BUILD DIPOLE–DIPOLE SURVEY
# ---------------------------

src_list = []
midpoints = []
separations = []

# electrode spacing (assumed uniform)
dx = x_electrodes[1] - x_electrodes[0]
a = a_factor * dx  # conceptual only here

# A at i, B at i+1 ; M at i+1+n_spacing, N at i+2+n_spacing
for iA in range(n_electrodes):
    iB = iA + 1
    iM = iA + 1 + n_spacing
    iN = iA + 2 + n_spacing

    if iN >= n_electrodes:
        break

    # 2D coordinates (x, z) for a 2D simulation
    A = np.r_[x_electrodes[iA], 0.0]
    B = np.r_[x_electrodes[iB], 0.0]
    M = np.r_[x_electrodes[iM], 0.0]
    N = np.r_[x_electrodes[iN], 0.0]

    rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
    src = dc.sources.Dipole([rx], A, B)
    src_list.append(src)

    midpoints.append(0.5 * (M[0] + N[0]))
    separations.append(abs(N[0] - M[0]))

midpoints = np.array(midpoints)
separations = np.array(separations)

survey = dc.Survey(src_list)

# <<< NEW: compute geometric factor K for apparent resistivity >>>
survey.set_geometric_factor(space_type="halfspace")

# ---------------------------
# 5) SIMULATION & FORWARD
# ---------------------------

# NOTE: depending on SimPEG version, change Simulation2DNodal to:
# - dc.Simulation2DNodal
# - or dc.Simulation2DCellCentered
# - or dc.Problem2D_CC (older API)
try:
    sim = dc.Simulation2DNodal(
        mesh=mesh,
        survey=survey,
        rhoMap=rho_map,
    )

    data = sim.dpred(rho_model)  # here "data" are apparent resistivities, since we set data_type + K
    ok = True
except Exception as e:
    ok = False
    st.error(f"Forward modelling failed: {e}")

# ---------------------------
# 6) DISPLAY: MODEL + PSEUDOSECTION
# ---------------------------

col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.subheader("Resistivity model (2D)")

    fig_m, ax_m = plt.subplots(figsize=(4, 5))
    m_img = ax_m.tripcolor(
        xc, zc, np.log10(rho_model),
        shading="gouraud"
    )
    ax_m.invert_yaxis()
    ax_m.set_xlabel("x (m)")
    ax_m.set_ylabel("z (m, mesh coords)")
    ax_m.set_title("log10(ρ) model")
    fig_m.colorbar(m_img, ax=ax_m, label="log10(ρ / Ω·m)")
    st.pyplot(fig_m, clear_figure=True)

    st.caption(
        "Background + rectangular anomaly. "
        "You can change ρ_bg, ρ_block and block size/position in the sidebar."
    )

with col2:
    st.subheader("Dipole–dipole apparent resistivity pseudosection")

    if ok:
        # simple pseudosection: x = midpoint; pseudo-depth = separation / 2
        pseudo_depth = separations / 2.0

        fig_d, ax_d = plt.subplots(figsize=(7, 5))
        sc = ax_d.scatter(
            midpoints,
            pseudo_depth,
            c=data,
            cmap="viridis",
            s=60,
            edgecolors="k"
        )
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Midpoint (m)")
        ax_d.set_ylabel("Pseudo-depth (m)")
        ax_d.set_title("Apparent resistivity (dipole–dipole)")
        fig_d.colorbar(sc, ax=ax_d, label="ρ_a (Ω·m)")
        ax_d.grid(True, linestyle=":", alpha=0.5)

        st.pyplot(fig_d, clear_figure=True)

        st.caption(
            "Pseudosection: each symbol corresponds to one dipole–dipole measurement. "
            "Vertical axis is a pseudo-depth (proportional to dipole separation), "
            "not a true inversion."
        )

# ---------------------------
# 7) NOTES
# ---------------------------

st.divider()
st.caption(
    "This app uses a 2D TensorMesh with a simple dipole–dipole line. "
    "We set `data_type='apparent_resistivity'` on the receiver and call "
    "`survey.set_geometric_factor()` so SimPEG converts voltages to ρₐ "
    "using the half-space geometric factor."
)
