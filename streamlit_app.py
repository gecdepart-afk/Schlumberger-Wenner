# ==============================
# 1D DC Forward Modelling (SimPEG)
# Streamlit app — Schlumberger + Wenner
# ==============================

# --- Core scientific libraries ---
import numpy as np                    # numerical arrays & math (efficient vector operations)
import pandas as pd                   # tabular data handling (for model table + CSV export)
import matplotlib.pyplot as plt       # plotting library for charts and model visualization
import streamlit as st                # Streamlit: web UI framework for Python (interactive apps)

# --- SimPEG modules for DC resistivity ---
from simpeg.electromagnetics.static import resistivity as dc  # SimPEG DC resistivity subpackage
from simpeg import maps               # “maps” connect model parameters to physical quantities

from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter

# ---------------------------
# 1) PAGE SETUP & HEADER
# ---------------------------

st.set_page_config(page_title="1D DC Forward (SimPEG)", page_icon="🪪", layout="wide")

st.title("1D DC Resistivity — Forward Modelling (Schlumberger vs Wenner)")
st.markdown(
    "Configure a layered Earth and **AB/2** geometry, then compute the **apparent resistivity** curves "
    "for both **Schlumberger** and **Wenner** arrays. "
    "Uses `simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers`."
)

# ==============================================================
# 2) SIDEBAR — INPUT PARAMETERS (geometry and layer model)
# ==============================================================

with st.sidebar:
    st.header("Geometry (AB/2 range)")

    # AB/2 min / max
    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input(
            "AB/2 min (m)", min_value=0.1, value=5.0, step=0.1, format="%.2f"
        )
    with colA2:
        ab2_max = st.number_input(
            "AB/2 max (m)", min_value=ab2_min + 0.1, value=300.0, step=1.0, format="%.2f"
        )

    n_stations = st.slider(
        "Number of stations", min_value=8, max_value=60, value=25, step=1
    )

    st.caption(
        "**Schlumberger:** MN/2 is set automatically to 10% of AB/2 "
        "(and clipped so MN/2 < 0.5·AB/2).  \n"
        "**Wenner:** AB = 3a, MN = a, centred at x = 0."
    )

    st.divider()
    st.header("Layers")

    # Number of layers
    n_layers = st.slider(
        "Number of layers", 3, 5, 4,
        help="Total layers (last layer is a half-space)."
    )

    # Default resistivities & thicknesses
    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers - 1)]

    # Resistivities
    layer_rhos = []
    for i in range(n_layers):
        layer_rhos.append(
            st.number_input(
                f"ρ Layer {i+1} (Ω·m)",
                min_value=0.1,
                value=float(default_rho[i]),
                step=0.1,
            )
        )

    # Thicknesses for first N−1 layers
    thicknesses = []
    if n_layers > 1:
        st.caption("Thicknesses for the **upper** N−1 layers (last layer is half-space):")
        for i in range(n_layers - 1):
            thicknesses.append(
                st.number_input(
                    f"Thickness L{i+1} (m)",
                    min_value=0.1,
                    value=float(default_thk[i]),
                    step=0.1,
                )
            )

# Convert thickness to NumPy array
thicknesses = np.r_[thicknesses] if len(thicknesses) else np.array([])

st.divider()

# ==============================================================
# 3) BUILD SURVEY GEOMETRY (Schlumberger + Wenner)
# ==============================================================

# AB/2 stations
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)

# Schlumberger: MN/2 = 0.1 * AB/2 (clipped)
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)

eps = 1e-6

# ---------------- Schlumberger survey ----------------
src_list_s = []
for L, a_s in zip(AB2, MN2):
    # Current electrodes A,B
    A_s = np.r_[-L, 0.0, 0.0]
    B_s = np.r_[+L, 0.0, 0.0]

    # Potential electrodes M,N near centre
    M_s = np.r_[-(a_s - eps), 0.0, 0.0]
    N_s = np.r_[+(a_s - eps), 0.0, 0.0]

    rx_s = dc.receivers.Dipole(M_s, N_s, data_type="apparent_resistivity")
    src_s = dc.sources.Dipole([rx_s], A_s, B_s)
    src_list_s.append(src_s)

survey_s = dc.Survey(src_list_s)

# ---------------- Wenner survey ----------------
# Wenner: A–M–N–B equally spaced by a.
# AB = 3a, AB/2 = 1.5a; we use AB/2 = L → a = (2/3)*L
src_list_w = []
for L in AB2:
    a_w = (2.0 / 3.0) * L

    A_w = np.r_[-1.5 * a_w, 0.0, 0.0]
    M_w = np.r_[-0.5 * a_w, 0.0, 0.0]
    N_w = np.r_[+0.5 * a_w, 0.0, 0.0]
    B_w = np.r_[+1.5 * a_w, 0.0, 0.0]

    rx_w = dc.receivers.Dipole(M_w, N_w, data_type="apparent_resistivity")
    src_w = dc.sources.Dipole([rx_w], A_w, B_w)
    src_list_w.append(src_w)

survey_w = dc.Survey(src_list_w)

# ==============================================================
# 4) SIMULATION & FORWARD MODELLING
# ==============================================================

rho = np.r_[layer_rhos]
rho_map = maps.IdentityMap(nP=len(rho))

# Schlumberger simulation
sim_s = dc.simulation_1d.Simulation1DLayers(
    survey=survey_s,
    rhoMap=rho_map,
    thicknesses=thicknesses,
)

# Wenner simulation
sim_w = dc.simulation_1d.Simulation1DLayers(
    survey=survey_w,
    rhoMap=rho_map,
    thicknesses=thicknesses,
)

try:
    rho_app_s = sim_s.dpred(rho)
    rho_app_w = sim_w.dpred(rho)
    ok = True
except Exception as e:
    ok = False
    st.error(f"Forward modelling failed: {e}")

# ==============================================================
# 5) DISPLAY RESULTS — curves, model, and data table
# ==============================================================

col1, col2 = st.columns([2, 1])

# --- LEFT: Apparent resistivity curves ---
with col1:
    st.subheader("Sounding curves (log–log)")
    if ok:
        fig, ax = plt.subplots(figsize=(7, 5))

        # Two arrays on same AB/2 axis
        ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger ρₐ")
        ax.loglog(AB2, rho_app_w, "s--", label="Wenner ρₐ")

        # Y-limits to full decades around both curves
        ymin = np.minimum(rho_app_s.min(), rho_app_w.min())
        ymax = np.maximum(rho_app_s.max(), rho_app_w.max())
        ymin = 10 ** np.floor(np.log10(ymin))
        ymax = 10 ** np.ceil(np.log10(ymax))
        ax.set_ylim(ymin, ymax)

        # Ticks only at decades
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="both", ls=":", alpha=0.7)

        ax.set_xlabel("AB/2 (m)")
        ax.set_ylabel("Apparent resistivity (Ω·m)")
        ax.set_title("Schlumberger vs Wenner — 1D VES (forward)")
        ax.legend()

        st.pyplot(fig, clear_figure=True)

        # Export both arrays in a single CSV
        df_out = pd.DataFrame({
            "AB/2 (m)": AB2,
            "MN/2 Schlumberger (m)": MN2,
            "ρa Schlumberger (Ω·m)": rho_app_s,
            "ρa Wenner (Ω·m)": rho_app_w,
        })
        st.download_button(
            "⬇️ Download synthetic data (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_VES_Schlumberger_Wenner.csv",
            mime="text/csv",
        )

# --- RIGHT: Layered model visualization ---
with col2:
    st.subheader("Layered model")
    if ok:
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        rho_vals = rho

        # Depth interfaces
        if len(thicknesses):
            interfaces = np.r_[0.0, np.cumsum(thicknesses)]
        else:
            interfaces = np.r_[0.0]

        z_bottom = interfaces[-1] + max(interfaces[-1] * 0.3, 10.0)

        tops = np.r_[interfaces, interfaces[-1]]
        bottoms = np.r_[interfaces[1:], z_bottom]

        for i in range(n_layers):
            ax2.fill_betweenx([tops[i], bottoms[i]], 0, rho_vals[i], alpha=0.35)
            ax2.text(
                rho_vals[i] * 1.05,
                (tops[i] + bottoms[i]) / 2,
                f"{rho_vals[i]:.1f} Ω·m",
                va="center",
                fontsize=9,
            )

        ax2.invert_yaxis()
        ax2.set_xlabel("Resistivity (Ω·m)")
        ax2.set_ylabel("Depth (m)")
        ax2.grid(True, ls=":")
        ax2.set_title("Block model")
        st.pyplot(fig2, clear_figure=True)

    # Model as table
    model_df = pd.DataFrame({
        "Layer": np.arange(1, n_layers + 1),
        "Resistivity (Ω·m)": rho,
        "Thickness (m)": [*thicknesses, np.nan],
        "Note": [""] * (n_layers - 1) + ["Half-space"],
    })
    st.dataframe(model_df, use_container_width=True)

# ==============================================================
# 6) FOOTNOTE — teaching notes
# ==============================================================

st.caption(
    "Notes: for Schlumberger, MN/2 is fixed to 10% of AB/2 (and clipped below 0.5·AB/2) to avoid "
    "numerical issues and electrode overlap. Wenner uses AB = 3a, MN = a, centred at x = 0. "
    "If you see instabilities at extreme geometries, reduce the AB/2 range or number of stations."
)
