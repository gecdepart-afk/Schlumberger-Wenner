# ==============================
# 1D DC Forward Modelling (SimPEG)
# Streamlit app – Schlumberger & Wenner + depth-of-investigation kernel
# ==============================

# --- Core scientific libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- SimPEG modules for DC resistivity ---
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

# ---------------------------
# 1) PAGE SETUP & HEADER
# ---------------------------

st.set_page_config(page_title="1D DC Forward (SimPEG)", page_icon="🪪", layout="wide")

st.title("1D DC Resistivity – Schlumberger & Wenner")
st.markdown(
    "Configure a layered Earth and **AB/2** geometry, then compute the **apparent resistivity** "
    "curves for **Schlumberger** and **Wenner** arrays.\n\n"
    "Uses `simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers`."
)

# ==============================================================
# 2) SIDEBAR – INPUT PARAMETERS (geometry and layer model)
# ==============================================================

with st.sidebar:
    st.header("Geometry (Schlumberger)")

    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input("AB/2 min (m)", min_value=0.1, value=5.0, step=0.1, format="%.2f")
    with colA2:
        ab2_max = st.number_input("AB/2 max (m)", min_value=ab2_min + 0.1, value=300.0, step=1.0, format="%.2f")

    n_stations = st.slider("Number of stations", min_value=8, max_value=60, value=25, step=1)

    st.caption("MN/2 (Schl.) is set automatically to 10% of AB/2 (and clipped to < 0.5·AB/2).")

    st.divider()
    st.header("Layers")

    n_layers = st.slider("Number of layers", 3, 5, 4, help="Total layers (last layer is a half-space).")

    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers - 1)]

    layer_rhos = []
    for i in range(n_layers):
        layer_rhos.append(
            st.number_input(f"ρ Layer {i+1} (Ω·m)", min_value=0.1, value=float(default_rho[i]), step=0.1)
        )

    thicknesses = []
    if n_layers > 1:
        st.caption("Thicknesses for the **upper** N−1 layers (last layer is half-space):")
        for i in range(n_layers - 1):
            thicknesses.append(
                st.number_input(f"Thickness L{i+1} (m)", min_value=0.1, value=float(default_thk[i]), step=0.1)
            )

# Convert thickness list to numpy array (SimPEG expects NumPy arrays, not Python lists)
thicknesses = np.r_[thicknesses] if len(thicknesses) else np.array([])

st.divider()

# ==============================================================
# 3) BUILD SURVEY GEOMETRY (AB/2, MN/2 positions)
# ==============================================================

AB2 = np.geomspace(ab2_min, ab2_max, n_stations)
MN2 = np.minimum(0.10 * AB2, 0.49 * AB2)   # Schlumberger MN/2
eps = 1e-6

# --- Schlumberger survey ---
src_list_s = []
for L, a in zip(AB2, MN2):
    A_s = np.r_[-L, 0.0, 0.0]
    B_s = np.r_[+L, 0.0, 0.0]
    M_s = np.r_[-(a - eps), 0.0, 0.0]
    N_s = np.r_[+(a - eps), 0.0, 0.0]

    rx_s = dc.receivers.Dipole(M_s, N_s, data_type="apparent_resistivity")
    src_s = dc.sources.Dipole([rx_s], A_s, B_s)
    src_list_s.append(src_s)

survey_s = dc.Survey(src_list_s)

# --- Wenner survey ---
src_list_w = []
for L in AB2:
    # AB = 2L ; Wenner spacing a = AB/3 = 2L/3
    a = (2.0 * L) / 3.0
    A_w = np.r_[-1.5 * a, 0.0, 0.0]
    M_w = np.r_[-0.5 * a, 0.0, 0.0]
    N_w = np.r_[+0.5 * a, 0.0, 0.0]
    B_w = np.r_[+1.5 * a, 0.0, 0.0]

    rx_w = dc.receivers.Dipole(M_w, N_w, data_type="apparent_resistivity")
    src_w = dc.sources.Dipole([rx_w], A_w, B_w)
    src_list_w.append(src_w)

survey_w = dc.Survey(src_list_w)

# ==============================================================
# 4) SIMULATION & FORWARD MODELLING
# ==============================================================

rho = np.r_[layer_rhos]
rho_map = maps.IdentityMap(nP=len(rho))

sim_s = dc.simulation_1d.Simulation1DLayers(
    survey=survey_s,
    rhoMap=rho_map,
    thicknesses=thicknesses,
)

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
# 4b) LAYER SENSITIVITY FOR DEPTH KERNEL
# ==============================================================

def compute_layer_sensitivity(sim, rho, station_index, rel_perturb=0.01):
    """
    Finite-difference sensitivity per layer, for a single datum.
    sens[j] = | d_i(rho_pert_j) - d_i(rho) |, normalized so max = 1.
    """
    base = sim.dpred(rho)
    n_layers = len(rho)
    sens = np.zeros(n_layers)

    for j in range(n_layers):
        rho_pert = rho.copy()
        rho_pert[j] *= (1.0 + rel_perturb)
        d_pert = sim.dpred(rho_pert)
        sens[j] = d_pert[station_index] - base[station_index]

    sens = np.abs(sens)
    if sens.max() > 0:
        sens = sens / sens.max()
    return sens


def build_vertical_kernel(sens_layers, thicknesses, n_layers, n_samples=400):
    """
    Expand layer sensitivities into a vertical depth kernel:
    - each layer sensitivity is taken constant within the layer
    - last layer is truncated at a plotting depth z_bottom
    Returns depth (z) and kernel(z).
    """
    if len(thicknesses):
        cum_thk = np.cumsum(thicknesses)           # len = n_layers-1
        interfaces = np.r_[0.0, cum_thk]           # len = n_layers
        z_last = interfaces[-1]
        z_bottom = z_last + max(z_last * 0.3, 10.0)
        tops_layers = interfaces
        bottoms_layers = np.r_[cum_thk, z_bottom]
    else:
        # single half-space
        tops_layers = np.array([0.0])
        bottoms_layers = np.array([10.0])
        z_bottom = 10.0

    depth = np.linspace(0.0, z_bottom, n_samples)
    kernel = np.zeros_like(depth)

    for i in range(n_layers):
        mask = (depth >= tops_layers[i]) & (depth <= bottoms_layers[i])
        kernel[mask] = sens_layers[i]

    if kernel.max() > 0:
        kernel = kernel / kernel.max()

    return depth, kernel

# ==============================================================
# 5) DISPLAY RESULTS – curves, model, and data table
# ==============================================================

col1, col2 = st.columns([2, 1])

# --- LEFT: Apparent resistivity curves ---
with col1:
    st.subheader("Sounding curves (log–log)")
    if ok:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(AB2, rho_app_s, "o-", label="Schlumberger")
        ax.loglog(AB2, rho_app_w, "s--", label="Wenner")
        ax.grid(True, which="both", ls=":")
        ax.set_xlabel("AB/2 (m)")
        ax.set_ylabel("Apparent resistivity (Ω·m)")
        ax.set_title("VES forward curves")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        df_out = pd.DataFrame({
            "AB/2 (m)": AB2,
            "MN/2 Schl (m)": MN2,
            "ρa Schl (ohm·m)": rho_app_s,
            "ρa Wenner (ohm·m)": rho_app_w,
        })
        st.download_button(
            "⬇️ Download synthetic data (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_VES_schl_wenner.csv",
            mime="text/csv",
        )

# --- RIGHT: Layered model visualization ---
with col2:
    st.subheader("Layered model")
    if ok:
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        rho_vals = rho

        if len(thicknesses):
            interfaces = np.r_[0.0, np.cumsum(thicknesses)]
        else:
            interfaces = np.r_[0.0]

        z_bottom = interfaces[-1] + max(interfaces[-1] * 0.3, 10.0)

        tops = np.r_[interfaces, interfaces[-1]]
        bottoms = np.r_[interfaces[1:], z_bottom]
        for i in range(n_layers):
            ax2.fill_betweenx([tops[i], bottoms[i]], 0, rho_vals[i], alpha=0.35)
            ax2.text(rho_vals[i] * 1.05, (tops[i] + bottoms[i]) / 2,
                     f"{rho_vals[i]:.1f} Ω·m", va="center", fontsize=9)

        ax2.invert_yaxis()
        ax2.set_xlabel("Resistivity (Ω·m)")
        ax2.set_ylabel("Depth (m)")
        ax2.grid(True, ls=":")
        ax2.set_title("Block model")
        st.pyplot(fig2, clear_figure=True)

    model_df = pd.DataFrame({
        "Layer": np.arange(1, n_layers + 1),
        "Resistivity (Ω·m)": rho,
        "Thickness (m)": [*thicknesses, np.nan],  # NaN for last layer (half-space)
        "Note": [""] * (n_layers - 1) + ["Half-space"]
    })
    st.dataframe(model_df, use_container_width=True)

# --------------------------------------------------------------
# DEPTH-OF-INVESTIGATION KERNEL (replaces old sensitivity plot)
# --------------------------------------------------------------
st.divider()
st.subheader("Depth-of-investigation kernel for a single datum")

if ok:
    colA, colB = st.columns(2)
    with colA:
        array_choice = st.selectbox("Array", ["Schlumberger", "Wenner"])
    with colB:
        idx_default = len(AB2) // 2
        station_index = st.slider(
            "Station index (0 = smallest AB/2)",
            0, len(AB2) - 1, idx_default,
        )

    selected_ab2 = AB2[station_index]
    st.caption(f"Selected station: #{station_index} – AB/2 = {selected_ab2:.2f} m")

    if array_choice == "Schlumberger":
        sim_current = sim_s
    else:
        sim_current = sim_w

    # 1) layer sensitivities for chosen datum
    sens_layers = compute_layer_sensitivity(sim_current, rho, station_index)

    # 2) expand into vertical kernel
    depth, kernel = build_vertical_kernel(sens_layers, thicknesses, n_layers, n_samples=400)

    # 3) effective depth and depth of maximum response
    if kernel.max() > 0:
        z_star = depth[np.argmax(kernel)]                    # depth of maximum sensitivity
        z_eff = np.sum(depth * kernel) / np.sum(kernel)      # effective depth
    else:
        z_star = 0.0
        z_eff = 0.0

    # 4) plot kernel with Z* and Z_E annotated
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.fill_betweenx(depth, 0, kernel, alpha=0.3, label="Sensitivity kernel")
    ax3.plot(kernel, depth, "-")

    # horizontal markers
    ax3.axhline(z_star, linestyle="--", color="C1", label="Z* (max sensitivity)")
    ax3.axhline(z_eff, linestyle=":", color="C2", label="Z_E (effective depth)")

    # small text labels on the right side
    x_text = 1.02 * kernel.max() if kernel.max() > 0 else 1.0
    ax3.text(x_text, z_star, "Z*", va="center", ha="left", color="C1")
    ax3.text(x_text, z_eff, "Z_E", va="center", ha="left", color="C2")

    ax3.invert_yaxis()
    ax3.set_xlabel("Relative sensitivity (normalised)")
    ax3.set_ylabel("Depth (m)")
    ax3.grid(True, ls=":")
    ax3.set_title(f"Depth kernel for datum #{station_index} ({array_choice})")
    ax3.legend(loc="upper right")
    st.pyplot(fig3, clear_figure=True)

    st.caption(
        f"For AB/2 = {selected_ab2:.2f} m:\n"
        f"• Z* (depth of maximum sensitivity) ≈ {z_star:.1f} m\n"
        f"• Z_E (effective depth of investigation) ≈ {z_eff:.1f} m\n\n"
        "The kernel shows how a single apparent-resistivity measurement is sensitive to depth. "
        "Z* marks the depth where the response is strongest; Z_E is the sensitivity-weighted "
        "average depth, often used as an effective investigation depth."
    )

# ==============================================================
# 6) FOOTNOTE – teaching notes
# ==============================================================

st.caption(
    "MN/2 is fixed to 10% of AB/2 for the Schlumberger array. "
    "Depth-of-investigation kernels here are simple approximations based on "
    "finite-difference layer sensitivities, expanded vertically within each layer."
)
