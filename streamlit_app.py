# ==============================
# 1D DC Forward Modelling (SimPEG)
# Streamlit app — Schlumberger, 2 models
# ==============================

# --- Core scientific libraries ---
import numpy as np                    # numerical arrays & math (efficient vector operations)
import pandas as pd                   # tabular data handling (for model table + CSV export)
import matplotlib.pyplot as plt       # plotting library for charts and model visualization
import streamlit as st                # Streamlit: web UI framework for Python (interactive apps)

# --- SimPEG modules for DC resistivity ---
from simpeg.electromagnetics.static import resistivity as dc  # SimPEG DC resistivity subpackage
from simpeg import maps               # “maps” connect model parameters to physical quantities

# ---------------------------
# 1) PAGE SETUP & HEADER
# ---------------------------

st.set_page_config(page_title="1D DC Forward (SimPEG)", page_icon="🪪", layout="wide")

st.title("1D DC Resistivity — Forward Modelling (Schlumberger)")
st.markdown(
    "Configure one or two layered Earth models and **AB/2** geometry, "
    "then compute the **apparent resistivity** curves. "
    "Uses `simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers`."
)

# ==============================================================
# 2) SIDEBAR — INPUT PARAMETERS (geometry and layer models)
# ==============================================================

with st.sidebar:
    st.header("Geometry (Schlumberger)")

    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input("AB/2 min (m)", min_value=0.1, value=5.0, step=0.1, format="%.2f")
    with colA2:
        ab2_max = st.number_input("AB/2 max (m)", min_value=ab2_min + 0.1, value=300.0, step=1.0, format="%.2f")

    n_stations = st.slider("Number of stations", min_value=8, max_value=60, value=25, step=1)

    st.caption("MN/2 is set automatically to 10% of AB/2 (and clipped to < 0.5·AB/2).")

    st.divider()
    st.header("Layers (both models use same number of layers)")

    n_layers = st.slider("Number of layers", 3, 5, 4,
                         help="Total layers (last layer is a half-space).")

    # common defaults
    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers - 1)]

    # ---------- Model 1 ----------
    st.subheader("Model 1")

    layer_rhos_1 = []
    for i in range(n_layers):
        layer_rhos_1.append(
            st.number_input(
                f"ρ Layer {i+1} (Ω·m) — M1",
                min_value=0.1,
                value=float(default_rho[i]),
                step=0.1
            )
        )

    thicknesses_1 = []
    if n_layers > 1:
        st.caption("Thicknesses for upper N−1 layers (Model 1):")
        for i in range(n_layers - 1):
            thicknesses_1.append(
                st.number_input(
                    f"Thickness L{i+1} (m) — M1",
                    min_value=0.1,
                    value=float(default_thk[i]),
                    step=0.1
                )
            )

    st.divider()

    # ---------- Model 2 ----------
    st.subheader("Model 2")

    # Use slightly different defaults for Model 2 just to encourage variation
    default_rho_2 = [val * 2 for val in default_rho]  # e.g. double resistivities
    default_thk_2 = default_thk[:]                    # same thickness defaults

    layer_rhos_2 = []
    for i in range(n_layers):
        layer_rhos_2.append(
            st.number_input(
                f"ρ Layer {i+1} (Ω·m) — M2",
                min_value=0.1,
                value=float(default_rho_2[i]),
                step=0.1
            )
        )

    thicknesses_2 = []
    if n_layers > 1:
        st.caption("Thicknesses for upper N−1 layers (Model 2):")
        for i in range(n_layers - 1):
            thicknesses_2.append(
                st.number_input(
                    f"Thickness L{i+1} (m) — M2",
                    min_value=0.1,
                    value=float(default_thk_2[i]),
                    step=0.1
                )
            )

# Convert thickness lists to numpy arrays
thicknesses_1 = np.r_[thicknesses_1] if len(thicknesses_1) else np.array([])
thicknesses_2 = np.r_[thicknesses_2] if len(thicknesses_2) else np.array([])

st.divider()

# ==============================================================
# 3) BUILD SURVEY GEOMETRY (AB/2, MN/2 positions)
# ==============================================================

AB2 = np.geomspace(ab2_min, ab2_max, n_stations)
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)
eps = 1e-6

src_list = []
for L, a in zip(AB2, MN2):
    A = np.r_[-L, 0.0, 0.0]
    B = np.r_[+L, 0.0, 0.0]
    M = np.r_[-(a - eps), 0.0, 0.0]
    N = np.r_[+(a - eps), 0.0, 0.0]

    rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
    src = dc.sources.Dipole([rx], A, B)
    src_list.append(src)

survey = dc.Survey(src_list)

# ==============================================================
# 4) SIMULATION & FORWARD MODELLING (2 models)
# ==============================================================

rho_1 = np.r_[layer_rhos_1]
rho_2 = np.r_[layer_rhos_2]

rho_map_1 = maps.IdentityMap(nP=len(rho_1))
rho_map_2 = maps.IdentityMap(nP=len(rho_2))

sim_1 = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    rhoMap=rho_map_1,
    thicknesses=thicknesses_1
)

sim_2 = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    rhoMap=rho_map_2,
    thicknesses=thicknesses_2
)

try:
    rho_app_1 = sim_1.dpred(rho_1)
    rho_app_2 = sim_2.dpred(rho_2)
    ok = True
except Exception as e:
    ok = False
    st.error(f"Forward modelling failed: {e}")

# ==============================================================
# 5) DISPLAY RESULTS — curves, models, and data table
# ==============================================================

col1, col2 = st.columns([2, 1])

from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter

# --- LEFT: Apparent resistivity curves ---
with col1:
    st.subheader("Sounding curves (log–log)")
    if ok:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.loglog(AB2, rho_app_1, "o-", label="Model 1")
        ax.loglog(AB2, rho_app_2, "s--", label="Model 2")

        # combine ranges of both models for nice frame
        ymin = min(rho_app_1.min(), rho_app_2.min())
        ymax = max(rho_app_1.max(), rho_app_2.max())
        ax.set_ylim(10**np.floor(np.log10(ymin)), 10**np.ceil(np.log10(ymax)))

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
        ax.set_title("Schlumberger VES (forward) — 2 models")
        ax.legend()

        st.pyplot(fig, clear_figure=True)

        # export both curves
        df_out = pd.DataFrame({
            "AB/2 (m)": AB2,
            "MN/2 (m)": MN2,
            "AppRes Model 1 (Ω·m)": rho_app_1,
            "AppRes Model 2 (Ω·m)": rho_app_2,
        })
        st.download_button(
            "⬇️ Download synthetic data (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_VES_two_models.csv",
            mime="text/csv",
        )

# --- RIGHT: Layered model visualization (for Model 1 only, to stay minimal) ---
with col2:
    st.subheader("Layered model (Model 1)")
    if ok:
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        rho_vals_1 = rho_1

        if len(thicknesses_1):
            interfaces_1 = np.r_[0.0, np.cumsum(thicknesses_1)]
        else:
            interfaces_1 = np.r_[0.0]

        z_bottom_1 = interfaces_1[-1] + max(interfaces_1[-1] * 0.3, 10.0)

        tops_1 = np.r_[interfaces_1, interfaces_1[-1]]
        bottoms_1 = np.r_[interfaces_1[1:], z_bottom_1]
        for i in range(n_layers):
            ax2.fill_betweenx([tops_1[i], bottoms_1[i]], 0, rho_vals_1[i], alpha=0.35)
            ax2.text(rho_vals_1[i] * 1.05, (tops_1[i] + bottoms_1[i]) / 2,
                     f"{rho_vals_1[i]:.1f} Ω·m", va="center", fontsize=9)

        ax2.invert_yaxis()
        ax2.set_xlabel("Resistivity (Ω·m)")
        ax2.set_ylabel("Depth (m)")
        ax2.grid(True, ls=":")
        ax2.set_title("Block model — Model 1")
        st.pyplot(fig2, clear_figure=True)

    # model table with both models side-by-side (same thickness pattern)
    model_df = pd.DataFrame({
        "Layer": np.arange(1, n_layers + 1),
        "ρ Model 1 (Ω·m)": rho_1,
        "ρ Model 2 (Ω·m)": rho_2,
        "Thickness (m)": [*thicknesses_1, np.nan],
        "Note": [""] * (n_layers - 1) + ["Half-space"]
    })
    st.dataframe(model_df, use_container_width=True)

# ==============================================================
# 6) FOOTNOTE — teaching notes
# ==============================================================

st.caption(
    "Notes: MN/2 is fixed to 10% of AB/2 (and clipped below 0.5·AB/2) to avoid numerical issues. "
    "If you see instabilities at extreme geometries, reduce AB/2 range."
)
