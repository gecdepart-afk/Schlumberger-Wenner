# ==============================
# 1D DC Forward Modelling (SimPEG)
# Streamlit app — Schlumberger only, user edits AB/2 only
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

# Configure the web app: title, icon, and layout
st.set_page_config(page_title="1D DC Forward (SimPEG)", page_icon="🪪", layout="wide")

# Title and description displayed at the top of the app
st.title("1D DC Resistivity — Forward Modelling (Schlumberger)")
st.markdown(
    "Configure a layered Earth and **AB/2** geometry, then compute the **apparent resistivity** curve. "
    "Uses `simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers`."
)

# ==============================================================
# 2) SIDEBAR — INPUT PARAMETERS (geometry and layer model)
# ==============================================================

with st.sidebar:   # everything inside here appears in the Streamlit sidebar
    st.header("Geometry (Schlumberger)")

    # --- AB/2 range controls (geometry setup) ---
    # Two side-by-side numeric inputs: minimum and maximum AB/2 electrode spacing
    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input("AB/2 min (m)", min_value=0.1, value=5.0, step=0.1, format="%.2f")
    with colA2:
        ab2_max = st.number_input("AB/2 max (m)", min_value=ab2_min + 0.1, value=300.0, step=1.0, format="%.2f")

    # Number of measurement stations between AB2_min and AB2_max
    n_stations = st.slider("Number of stations", min_value=8, max_value=60, value=25, step=1)

    # Fix MN/2 to a standard ratio (10% of AB/2)
    st.caption("MN/2 is set automatically to 10% of AB/2 (and clipped to < 0.5·AB/2).")

    st.divider()
    st.header("Layers")

    # --- Layer parameters (resistivity + thickness) ---
    # Choose number of layers (3–5). The last one is the infinite half-space.
    n_layers = st.slider("Number of layers", 3, 5, 4, help="Total layers (last layer is a half-space).")

    # Default layer resistivities and thicknesses (editable)
    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers - 1)]

    # Resistivity input per layer
    # ---------- Model 1 ----------
    st.subheader("Model 1")
    layer_rhos_1 = []
    for i in range(n_layers):
        layer_rhos_1.append(
            st.number_input(f"ρ Layer {i+1} (Ω·m)", min_value=0.1, value=float(default_rho_1[i]), step=0.1)
        )

    # Thickness input for the top N−1 layers (the last layer has infinite thickness)
    thicknesses_1 = []
    if n_layers > 1:
        st.caption("Thicknesses for the **upper** N−1 layers (last layer is half-space):")
        for i in range(n_layers - 1):
            thicknesses_1.append(
                st.number_input(f"Thickness L{i+1} (m)", min_value=0.1, value=float(default_thk_1[i]), step=0.1)
            )
    st.divider
    st.subheader("Model 2")
    layer_rhos_2 = []
    for i in range(n_layers):
        layer_rhos_2.append(
            st.number_input(f"ρ Layer {i+1} (Ω·m)", min_value=0.1, value=float(default_rho_2[i]), step=0.1)
        )

    # Thickness input for the top N−1 layers (the last layer has infinite thickness)
    thicknesses_2 = []
    if n_layers > 1:
        st.caption("Thicknesses for the **upper** N−1 layers (last layer is half-space):")
        for i in range(n_layers - 1):
            thicknesses_2.append(
                st.number_input(f"Thickness L{i+1} (m)", min_value=0.1, value=float(default_thk_2[i]), step=0.1)
            )

# Convert thickness list to numpy array (SimPEG expects NumPy arrays, not Python lists)
thicknesses_1 = np.r_[thicknesses_1] if len(thicknesses_1) else np.array([])
thicknesses_2 = np.r_[thicknesses_2] if len(thicknesses_2) else np.array([])
st.divider()

# ==============================================================
# 3) BUILD SURVEY GEOMETRY (AB/2, MN/2 positions)
# ==============================================================

# Create logarithmically spaced AB/2 electrode spacings (common in field surveys)
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)

# Define MN/2 spacing automatically (10% of AB/2, limited to avoid overlap)
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)

# Small offset epsilon avoids the situation where M = A or N = B (which breaks math)
eps = 1e-6

# Prepare SimPEG “sources” — one per station
# Each source defines:
#  - A and B (current electrodes)
#  - one receiver (M–N dipole measuring potential)
src_list = []
for L, a in zip(AB2, MN2):
    # Positions of electrodes along the x-axis (y,z = 0)
    A = np.r_[-L, 0.0, 0.0]
    B = np.r_[ +L, 0.0, 0.0]
    M = np.r_[ -(a - eps), 0.0, 0.0]
    N = np.r_[ +(a - eps), 0.0, 0.0]

    # Receiver measures apparent resistivity directly
    rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")

    # Source = AB current dipole carrying this receiver
    src = dc.sources.Dipole([rx], A, B)
    src_list.append(src)

# Create the SimPEG survey object from all sources
survey = dc.Survey(src_list)

# ==============================================================
# 4) SIMULATION & FORWARD MODELLING
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
# 5) DISPLAY RESULTS — curve, model, and data table
# ==============================================================

col1, col2 = st.columns([2, 1])  # layout: wide chart + narrow model panel

# --- LEFT: Apparent resistivity curve ---
with col1:
    st.subheader("Sounding curve (log–log)")
    if ok:
        # Create a figure using matplotlib
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(AB2, rho_app_1, "o-", label="ρₐ (predicted)")
        ax.loglog(AB2, rho_app_2, "o-", label="ρₐ (predicted)")
        ax.grid(True, which="both", ls=":")
        ax.set_xlabel("AB/2 (m)")
        ax.set_ylabel("Apparent resistivity (Ω·m)")
        ax.set_title("Schlumberger VES (forward)")
        ax.legend()

        # Show it inside Streamlit
        st.pyplot(fig, clear_figure=True)

        # Export results as CSV for external plotting (Excel, Python, etc.)
        df_out = pd.DataFrame({
            "AB/2 (m)": AB2,
            "MN/2 (m)": MN2,
            "Apparent resistivity (ohm·m)": rho_app,
        })
        st.download_button(
            "⬇️ Download synthetic data (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_VES.csv",
            mime="text/csv",
        )

# --- RIGHT: Layered model visualization ---
with col2:
    st.subheader("Layered model")
    if ok:
        # “Block model” diagram: resistivity vs. depth
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        rho_vals = rho

        # Compute depth interfaces from thicknesses
        if len(thicknesses):
            interfaces = np.r_[0.0, np.cumsum(thicknesses)]
        else:
            interfaces = np.r_[0.0]

        # Add bottom depth for plotting last half-space
        z_bottom = interfaces[-1] + max(interfaces[-1] * 0.3, 10.0)

        # Draw one filled rectangle per layer
        tops = np.r_[interfaces, interfaces[-1]]
        bottoms = np.r_[interfaces[1:], z_bottom]
        for i in range(n_layers):
            ax2.fill_betweenx([tops[i], bottoms[i]], 0, rho_vals[i], alpha=0.35)
            ax2.text(rho_vals[i] * 1.05, (tops[i] + bottoms[i]) / 2,
                     f"{rho_vals[i]:.1f} Ω·m", va="center", fontsize=9)

        ax2.invert_yaxis()               # depth increases downward
        ax2.set_xlabel("Resistivity (Ω·m)")
        ax2.set_ylabel("Depth (m)")
        ax2.grid(True, ls=":")
        ax2.set_title("Block model")
        st.pyplot(fig2, clear_figure=True)

    # Display the same model as a table
    model_df = pd.DataFrame({
        "Layer": np.arange(1, n_layers + 1),
        "Resistivity (Ω·m)": rho,
        "Thickness (m)": [*thicknesses, np.nan],  # NaN for last layer (half-space)
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
