"""
Logistic Map Simulator - Streamlit Web Application v1.0
Ported from MATLAB App Designer (v2.0)
Author: Altug Aksoy (adapted for Python/Streamlit)
Citation: Aksoy et al., Chaos, 2024

Finalized: 9 January 2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from io import BytesIO
import base64
from datetime import datetime
import pandas as pd
from logistic_map_simulator_v9 import LogisticMapSimulator
from streamlit_js_eval import streamlit_js_eval
from sim_data import PRECALC_DATA
from scipy.stats import gaussian_kde


# === SIMPLE RESPONSIVE FIGURE SIZING ===
def is_mobile_layout():
    """Detect mobile layout based on viewport width from JavaScript."""
    vw = st.session_state.get("viewport_width", None)
    if vw is None:
        # Fallback: JS hasn't evaluated yet or is unavailable. Assume desktop for first load.
        return False
    return vw < 768  # Standard mobile breakpoint (tablet+phone)


# === GLOBAL FIGURE SIZE HELPER ===
def get_plot_figsize():
    if is_mobile_layout():
        # Slightly narrower but still readable; plots stack vertically
        return (5, 4)
    else:
        # 4 dynamics plots (2×2 grid)
        return (7, 6)

def get_bif_figsize():
    # Single bifurcation plot
    if is_mobile_layout():
        return (6, 4.5)
    else:
        return (10, 6)


# === CUSTOM CSS FOR TAB BUTTON STYLING ===
# 1. Base CSS (Always applied: Top Tabs, Mobile Layouts)
st.markdown("""
<style>
    /* Tab button styling with black borders (Top Navigation) */
    div[data-testid="column"] button {
        width: 100%;
        border-radius: 8px;
        padding: 10px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        border: 2px solid #000000 !important;
    }
    
    /* Unselected tab button (Top Nav) */
    div[data-testid="column"] button[kind="secondary"] {
        background-color: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Selected tab button (Top Nav) */
    div[data-testid="column"] button[kind="primary"] {
        background: linear-gradient(135deg, #32b8c6 0%, #1d7480 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(50, 184, 198, 0.3);
    }
    
    /* Hover effects */
    div[data-testid="column"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(50, 184, 198, 0.4);
    }
    
    /* Tab row divider line */
    [data-testid="stHorizontalBlock"]:has(button[kind="primary"], button[kind="secondary"]) {
        border-bottom: 2px solid #000000;
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    
    /* Keep sidebar visible */
    [data-testid="stSidebar"] {
        display: block !important;
    }

    /* === MOBILE LAYOUT TWEAKS === */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] button[key^="tab_btn_"] {
            margin-bottom: 0.35rem;
        }
        section.main > div {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        html, body {
            font-size: 16px !important;
        }
    }
            
    /* === GLOBAL SIDEBAR LABEL STYLING === */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 16px !important;
    }
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Dynamic Sidebar Button Styling (Depends on Active Tab)
# If we are in Tabs 0, 1, or 2: Primary Button is RED (Alert/Run)
# If we are in Tab 3 (Info): Primary Button is GREEN (Selected/Nav)
current_tab = st.session_state.get('selected_tab_index', 0)

if current_tab in [0, 1, 2, 3]:
    # === RED WARNING STYLE FOR RUN BUTTONS ===
    st.markdown("""
    <style>
        /* PRIMARY (Settings Changed/Run Needed): White Background, RED Text/Border */
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #ffffff !important;
            color: #ff4b4b !important;
            border: 3px solid #ff4b4b !important;
            font-weight: 600;
        }
        /* SECONDARY (Simulation Ran): Green Background, White Text */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #228B22 !important;
            color: white !important;
            border: 3px solid #228B22 !important;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # === GREEN NAVIGATION STYLE FOR INFO BUTTONS ===
    st.markdown("""
    <style>
        /* PRIMARY (Selected): White Background, GREEN Text/Border */
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #ffffff !important;
            color: #228B22 !important;
            border: 3px solid #228B22 !important;
            font-weight: 600;
        }
        /* SECONDARY (Unselected): Green Background, White Text */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #228B22 !important;
            color: white !important;
            border: 3px solid #228B22 !important;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Logistic Map Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)


# === SIDEBAR MINIMUM WIDTH ===
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 320px !important;
    }
</style>
""", unsafe_allow_html=True)


# === SESSION STATE INITIALIZATION ===
if 'simulation_ran' not in st.session_state:
    st.session_state.simulation_ran = False
if 'selected_tab_index' not in st.session_state:
    st.session_state.selected_tab_index = 0
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = None
if 'button_color' not in st.session_state:
    st.session_state.button_color = 'normal'
if 'bif_button_color' not in st.session_state:
    st.session_state.bif_button_color = 'normal'
if 'bif_last_params' not in st.session_state:
    st.session_state.bif_last_params = None
if 'bifurcation_computed' not in st.session_state:
    st.session_state.bifurcation_computed = False
if 'iter_diff_value' not in st.session_state:
    st.session_state.iter_diff_value = 1
if 'pred_button_color' not in st.session_state:
    st.session_state.pred_button_color = 'normal'
if 'pred_ensemble_metric' not in st.session_state:
    st.session_state.pred_ensemble_metric = 'median'
if 'last_selected_r_indices' not in st.session_state:
    st.session_state.last_selected_r_indices = []
if 'last_selected_mb_indices' not in st.session_state:
    st.session_state.last_selected_mb_indices = []
if 'pred_last_params' not in st.session_state:
    st.session_state.pred_last_params = None
if 'ens_button_color' not in st.session_state:
    st.session_state.ens_button_color = 'normal'
if 'ens_plot_clicked' not in st.session_state:
    st.session_state.ens_plot_clicked = False
if 'ens_last_params' not in st.session_state:
    st.session_state.ens_last_params = None
if 'fig4_ran' not in st.session_state:
    st.session_state.fig4_ran = False
if 'fig4_data' not in st.session_state:
    st.session_state.fig4_data = None
if 'ens_spread_type' not in st.session_state:
    st.session_state.ens_spread_type = "10th-90th Percentiles"
if 'info_sub_tab' not in st.session_state:
    st.session_state.info_sub_tab = "intro"  # Options: 'intro', 'usage', 'about'


# === Initialize Checkbox States and Tracker ===
if 'viz_show_mean' not in st.session_state: st.session_state.viz_show_mean = True
if 'viz_show_median' not in st.session_state: st.session_state.viz_show_median = False
if 'viz_show_mode' not in st.session_state: st.session_state.viz_show_mode = False
if 'last_central_stat' not in st.session_state: st.session_state.last_central_stat = "Mean"


# === VIEWPORT WIDTH DETECTION (JS → Python) ===
if 'viewport_width' not in st.session_state:
    try:
        vw = streamlit_js_eval(
            js_expressions="window.innerWidth",
            key="viewport_width_js",
            want_output=True,
        )
        if vw is not None:
            st.session_state.viewport_width = vw
    except Exception as e:
        # Graceful fallback if JS eval fails
        st.session_state.viewport_width = None


# === INITIALIZE SIMULATOR ===
@st.cache_resource
def get_simulator():
    return LogisticMapSimulator()

simulator = get_simulator()


# === SESSION STATE FOR PREDICTABILITY TAB ===
if 'pred_use_precalc' not in st.session_state:
    st.session_state.pred_use_precalc = True
if 'pred_data' not in st.session_state:
    st.session_state.pred_data = PRECALC_DATA.copy()
if 'pred_calculation_seed' not in st.session_state:
    st.session_state.pred_calculation_seed = 42
if 'pred_ensemble_metric' not in st.session_state:
    st.session_state.pred_ensemble_metric = 'median'


# === HELPER FUNCTION: CREATE WHITE-BASED COLORMAP ===
def create_white_based_colormap(base_cmap_name):
    """Create a custom colormap that starts with white for the lowest values."""
    base_cmap = plt.get_cmap(base_cmap_name)
    n_colors = 256
    base_colors = base_cmap(np.linspace(0, 1, n_colors))
    new_colors = np.ones((n_colors, 4))
    transition_idx = int(n_colors * 0.1)
    
    for i in range(transition_idx):
        alpha = i / transition_idx
        new_colors[i] = (1 - alpha) * np.array([1, 1, 1, 1]) + alpha * base_colors[0]
    
    new_colors[transition_idx:] = base_colors[:(n_colors - transition_idx)]
    custom_cmap = LinearSegmentedColormap.from_list(f'white_{base_cmap_name}', new_colors)
    return custom_cmap


# === TITLE AND HEADER ===
st.markdown("<h1 style='text-align: center; color: #32b8c6;'>🦋 Logistic Map Simulator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: #888;'>Exploration of chaos, bifurcations, and predictability in the logistic map</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px; color: #666; font-style: italic;'>Based on: Aksoy, A. (2024). Chaos, 34, 011102.</p>", unsafe_allow_html=True)
st.markdown("---")


# === TAB NAVIGATION ===
tab_col_space1, tab_col1, tab_col2, tab_col3, tab_col4, tab_col5, tab_col_space2 = st.columns([0.05, 1, 1, 1, 1, 1, 0.05])

# Tab 0: Bifurcation
with tab_col1:
    btn1_clicked = st.button("Bifurcation", type="primary" if st.session_state.selected_tab_index == 0 else "secondary", width='stretch', key="tab_btn_0")
    if btn1_clicked and st.session_state.selected_tab_index != 0:
        st.session_state.selected_tab_index = 0
        st.rerun()

# Tab 1: Dynamics
with tab_col2:
    btn2_clicked = st.button("Dynamics", type="primary" if st.session_state.selected_tab_index == 1 else "secondary", width='stretch', key="tab_btn_1")
    if btn2_clicked and st.session_state.selected_tab_index != 1:
        st.session_state.selected_tab_index = 1
        st.rerun()

# Tab 2: Predictability
with tab_col3:
    btn3_clicked = st.button("Predictability", type="primary" if st.session_state.selected_tab_index == 2 else "secondary", width='stretch', key="tab_btn_2")
    if btn3_clicked and st.session_state.selected_tab_index != 2:
        st.session_state.selected_tab_index = 2
        st.rerun()

# Tab 3: Comparative Analysis
with tab_col4:
    if st.button("Comparative Error Growth", type="primary" if st.session_state.selected_tab_index == 3 else "secondary", width='stretch', key="tab_btn_3"):
        st.session_state.selected_tab_index = 3
        st.rerun()

# Tab 4: Info
with tab_col5:
    if st.button("Info", type="primary" if st.session_state.selected_tab_index == 4 else "secondary", width='stretch', key="tab_btn_4"):
        st.session_state.selected_tab_index = 4
        st.rerun()

selected_tab = st.session_state.selected_tab_index


# === SIDEBAR CONFIGURATION ===
with st.sidebar:
    if selected_tab == 0:
        with st.container(border=True):
            st.markdown("#### Bifurcation Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                r_min_bif = st.number_input("r min", value=2.5, step=0.1, format="%.2f")
            with col2:
                r_max_bif = st.number_input("r max", value=4.0, step=0.1, format="%.2f")
            
            x0_bif = st.slider("x₀ (initial condition)", 0.0, 1.0, value=0.5, step=0.01)
            transient_iters = st.number_input("Transient Iterations to Skip", value=200, min_value=50, max_value=500, step=50)
            plot_iters = st.number_input("Plot Number of Iteration Ater Transient", value=1000, min_value=500, max_value=5000, step=100)
        
        with st.container(border=True):
            st.markdown("#### Resolution & Display")
            
            resolution = st.number_input("Grid Resolution", value=1000, min_value=500, max_value=5000, step=100)

            show_density = True
            use_power_scale = True
            
            colormap = st.selectbox("Colormap", ["turbo", "jet", "rainbow"], index=0)

            gamma_value = st.slider(
                "Gamma (Enhance low-density features)", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.2, 
                step=0.05,
                help="Adjusts the color contrast. Lower values (like 0.2) allow you to see the full range of densities, including very faint structures at smaller values."
            )
        
        bif_current_params = {
            'r_min': r_min_bif, 'r_max': r_max_bif, 'x0': x0_bif,
            'transient': transient_iters, 'plot': plot_iters, 'resolution': resolution,
            'density': show_density, 'colormap': colormap, 'power_scale': use_power_scale,
            'gamma': gamma_value
        }

        if st.session_state.bif_last_params != bif_current_params:
            st.session_state.bif_button_color = 'normal'
            st.session_state.bifurcation_computed = False
        
        bif_button_type = 'secondary' if st.session_state.bif_button_color == 'success' else 'primary'
        run_bif_button = st.button("▶️ Compute Bifurcation", type=bif_button_type, width='stretch')
        
        if run_bif_button:
            with st.spinner("Computing bifurcation diagram..."):
                if show_density:
                    bifurcation_data = simulator.compute_bifurcation_diagram_with_density(
                        r_min=r_min_bif, r_max=r_max_bif, num_r=resolution,
                        x_min=0.0, x_max=1.0, num_x=resolution,
                        num_iterations=plot_iters, iterations_discard=transient_iters
                    )
                else:
                    bifurcation_data = simulator.compute_bifurcation_diagram(
                        r_min=r_min_bif, r_max=r_max_bif, num_r=resolution,
                        x_min=0.0, x_max=1.0, num_x=resolution,
                        num_iterations=plot_iters, iterations_discard=transient_iters
                    )
                
                st.session_state.bifurcation_data = bifurcation_data
                st.session_state.bifurcation_params_used = bif_current_params.copy()
                st.session_state.bifurcation_computed = True
                st.session_state.bif_last_params = bif_current_params
                st.session_state.bif_button_color = 'success'
                st.session_state.bif_figure_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()


    elif selected_tab == 1:
        with st.container(border=True):
            st.markdown("### Dynamical Regime Constraint for Parameter r")
        
            regime = st.selectbox(
                "Dynamical Regime",
                ["Chaotic", "Deterministic (Single-Valued)", "Deterministic (Periodic)"],
                help="Select the dynamical regime to explore",
                label_visibility="collapsed"
            )
        
            defaults = simulator.REGIME_DEFAULTS[regime]
        
        with st.container(border=True):
            st.markdown("### System Parameters (Truth and Modeled)")
            
            plus_style = "<div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: -8px;'>+</div>"
            
            def param_label(text):
                st.markdown(f"<div style='font-size: 16px; padding-bottom: 12px;'>{text}</div>", unsafe_allow_html=True)
            
            # --- r (truth) ---
            param_label("r (truth)")
            col_r1, col_r2, col_r3 = st.columns([0.55, 0.10, 0.35], gap="small", vertical_alignment="center")
            with col_r1:
                r_true_slider = st.slider("r_truth_hidden", float(defaults['param_slider_limits'][0]), float(defaults['param_slider_limits'][1]), float(defaults['param_slider_value']), 0.01, key="r_true_slider", label_visibility="collapsed")
            with col_r2: st.markdown(plus_style, unsafe_allow_html=True)
            with col_r3: r_true_adj = st.number_input("adj", 0.0, format="%.2e", step=0.0, key="r_true_adj", label_visibility="collapsed")
            r_true = r_true_slider + r_true_adj
            
            # --- x0 (truth) ---
            param_label("x₀ (truth)")
            col_x1, col_x2, col_x3 = st.columns([0.55, 0.10, 0.35], gap="small", vertical_alignment="center")
            with col_x1:
                x0_true_slider = st.slider("x0_truth_hidden", float(defaults['init_slider_limits'][0]), float(defaults['init_slider_limits'][1]), float(defaults['init_slider_value']), 0.01, key="x0_true_slider", label_visibility="collapsed")
            with col_x2: st.markdown(plus_style, unsafe_allow_html=True)
            with col_x3: x0_true_adj = st.number_input("adj", 0.0, format="%.2e", step=0.0, key="x0_true_adj", label_visibility="collapsed")
            x0_true = x0_true_slider + x0_true_adj
            
            # --- r (model) ---
            param_label("r (model)")
            col_rm1, col_rm2, col_rm3 = st.columns([0.55, 0.10, 0.35], gap="small", vertical_alignment="center")
            with col_rm1:
                r_model_slider = st.slider("r_model_hidden", float(defaults['param_slider_limits'][0]), float(defaults['param_slider_limits'][1]), float(defaults['param_slider_value']), 0.01, key="r_model_slider", label_visibility="collapsed")
            with col_rm2: st.markdown(plus_style, unsafe_allow_html=True)
            with col_rm3: r_model_adj = st.number_input("adj", 0.0, format="%.2e", step=0.0, key="r_model_adj", label_visibility="collapsed")
            r_model = r_model_slider + r_model_adj
            
            # --- x0 (model) ---
            param_label("x₀ (model)")
            col_xm1, col_xm2, col_xm3 = st.columns([0.55, 0.10, 0.35], gap="small", vertical_alignment="center")
            with col_xm1:
                x0_model_slider = st.slider("x0_model_hidden", float(defaults['init_slider_limits'][0]), float(defaults['init_slider_limits'][1]), float(defaults['init_slider_value']), 0.01, key="x0_model_slider", label_visibility="collapsed")
            with col_xm2: st.markdown(plus_style, unsafe_allow_html=True)
            with col_xm3: x0_model_adj = st.number_input("adj", 1e-5, format="%.2e", step=0.0, key="x0_model_adj", label_visibility="collapsed")
            x0_model = x0_model_slider + x0_model_adj
            
            st.markdown("") 
            num_steps = st.number_input("Number of Simulation Iterations/Steps (n)", 10, 1000, 100, 10)
        
        with st.container(border=True):
            st.markdown("### Predictability Limit")
            col_pt1, col_pt2, col_pt3 = st.columns([0.34, 0.12, 0.34], gap="small", vertical_alignment="center")
            with col_pt1: pred_mantissa = st.number_input("Assumed Limit", 1.0, 9.999, 1.0, 0.1, format="%.3f", key="pred_mantissa")
            with col_pt2: st.markdown("<div style='padding-top: 8px; text-align: center; font-size: 15px; font-weight: bold; '>x 10^</div>", unsafe_allow_html=True)
            with col_pt3: 
                st.markdown("<div style='padding-top: 25px;'></div>", unsafe_allow_html=True)
                pred_exponent = st.number_input("Exponent", -6, 0, -1, 1, key="pred_exponent", label_visibility="collapsed")
            pred_thresh = pred_mantissa * (10 ** pred_exponent)

        with st.container(border=True):
            st.markdown("### Ensemble Settings")
            use_ensemble = st.toggle("Enable Ensemble Simulation", value=False)
            
            if use_ensemble:
                ens_size = st.number_input("Ensemble Size", 2, 500, 50, 1)
                
                # 1. Primary Statistic Selection
                central_stat = st.selectbox("Primary Statistic (for Main Plots)", ["Mean", "Median", "Mode"], index=0)

                # === NEW: Spread Selection Switch ===
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                ens_spread_type = st.radio(
                    "Ensemble Range Display",
                    ["10th-90th Percentiles", "Min-Max Range"],
                    index=0,
                    help="Choose between the statistical variability (10th-90th) or the absolute full range (Min-Max) of the ensemble.",
                    key="ens_spread_type"
                )

                # === STATE MANAGEMENT ===
                # Update Session State DIRECTLY before rendering widgets.
                # This forces the active metric to be True.
                # Because we update the state here, we MUST NOT pass 'value=' to the checkboxes below.
                if central_stat == 'Mean': 
                    st.session_state.viz_show_mean = True
                elif central_stat == 'Median': 
                    st.session_state.viz_show_median = True
                elif central_stat == 'Mode': 
                    st.session_state.viz_show_mode = True

                initval_pert = st.number_input("Initial Value Perturbation", 1e-10, 0.1, 1e-4, format="%.2e")
                param_pert = st.number_input("Parameter Perturbation", 0.0, 0.1, 0.0, format="%.2e")
                
                st.markdown("---")
                st.markdown("**Detailed Analysis Settings**")
                
                # Sliders (1-based indexing)
                st.markdown("<span style='font-size: 14px;'>Select Time Steps for Histograms:</span>", unsafe_allow_html=True)
                max_steps = num_steps 
                hist_t1 = st.slider("First Time Step Displayed", 1, max_steps, min(5, max_steps), key="hist_t1")
                hist_t2 = st.slider("Second Time Step Displayed", 1, max_steps, min(20, max_steps), key="hist_t2")
                hist_t3 = st.slider("Third Time Step Displayed", 1, max_steps, min(60, max_steps), key="hist_t3")

                st.markdown("<span style='font-size: 14px;'>Metrics to Overlay/Compare:</span>", unsafe_allow_html=True)
                
                # Adjusted column ratio
                col_m1, col_m2 = st.columns([0.35, 0.65])
                
                # 2. Checkboxes
                # We do NOT pass 'value='. Streamlit reads the state we set above automatically.
                with col_m1:
                    show_ens_mean = st.checkbox("Ens. Mean", key="viz_show_mean", 
                                              disabled=(central_stat == 'Mean'))
                    
                    show_ens_median = st.checkbox("Ens. Median", key="viz_show_median", 
                                                disabled=(central_stat == 'Median'))
                    
                    show_ens_mode = st.checkbox("Ens. Mode", key="viz_show_mode", 
                                              disabled=(central_stat == 'Mode'))
                    
                with col_m2:
                    show_traj_mean = st.checkbox("Det. Traj. from Init. Mean", value=True, key="viz_show_traj_mean")
                    show_traj_median = st.checkbox("Det. Traj. from Init. Median", value=False, key="viz_show_traj_median")
                    show_traj_mode = st.checkbox("Det. Traj. from Init. Mode", value=False, key="viz_show_traj_mode")

                # KDE Toggle
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("<span style='font-size: 14px;'>Plot the Kernel Density Estimate?</span>", unsafe_allow_html=True)
                show_kde_option = st.radio(
                    "Show KDE",
                    ["Yes", "No"],
                    index=0,  # Default to Yes
                    horizontal=True,
                    label_visibility="collapsed",
                    key="viz_show_kde"
                )

            else:
                # Default values if ensemble is off
                ens_size = 1; initval_pert = 0.0; param_pert = 0.0; central_stat = "Mean"
                hist_t1, hist_t2, hist_t3 = 10, 30, 60
                # Initialize state if missing to prevent errors
                if 'viz_show_mean' not in st.session_state: st.session_state.viz_show_mean = True
        
        with st.container(border=True):
            st.markdown("#### State-Space Settings")
            iter_diff = st.number_input("Lag Parameter", 1, 50, 1, 1)

        # Visual settings for main plots
        show_time_series = True
        show_time_series_diff = True
        show_state_space = True
        show_state_space_diff = True
        show_ensemble_spread = use_ensemble
        
        # === PARAMETER PACKAGING ===
        # 1. Simulation Parameters (Changes turn button RED)
        sim_params = {
            'r_true': r_true, 
            'r_model': r_model, 
            'x0_true': x0_true, 
            'x0_model': x0_model,
            'num_steps': num_steps, 
            'use_ensemble': use_ensemble, 
            'ens_size': ens_size,
            'init_val_pert': initval_pert if use_ensemble else 0.0,
            'param_pert': param_pert if use_ensemble else 0.0
        }
        
        # Initialize last_sim_params if not present
        if 'last_sim_params' not in st.session_state:
            st.session_state.last_sim_params = None
        
        # Button Logic: Only turn RED if SIMULATION parameters change
        if st.session_state.last_sim_params != sim_params:
            st.session_state.button_color = 'normal'
        
        button_type = 'secondary' if st.session_state.button_color == 'success' else 'primary'
        
        # Run Button
        run_button = st.button("▶️ Run Simulation", type=button_type, width='stretch', key="run_sim_main")
        
        if run_button:
            with st.spinner('Running simulation...'):
                st.session_state.results = simulator.run_simulation(
                    r_true=r_true, x0_true=x0_true, r_model=r_model, x0_model=x0_model,
                    num_steps=num_steps, pred_thresh=pred_thresh, ensemble_enabled=use_ensemble,
                    ensemble_size=ens_size, init_val_pert=initval_pert, param_pert=param_pert,
                    ensemble_stat=central_stat
                )
                st.session_state.simulation_ran = True
                st.session_state.last_sim_params = sim_params
                st.session_state.button_color = 'success'
                st.session_state.figure_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()


    elif selected_tab == 2:
        with st.container(border=True):
            st.markdown("### Predictability Analysis Controls")

            cache = st.session_state.pred_data
            r_values = cache['r_values']
            model_bias_values = cache['model_bias_values']

            # Ensemble Metric Selection
            ensemble_metric = st.selectbox(
                "Ensemble Metric",
                ["mean", "median", "mode"],
                index=["mean", "median", "mode"].index(st.session_state.pred_ensemble_metric),
                help="Metric used to represent ensemble predictions",
                key="ensemble_metric_pred"
            )

            # Helper for consistent label styling (14px, non-bold)
            def section_label(text):
                st.markdown(f"<p style='font-size: 16px; margin-bottom: 0px;'>{text}</p>", unsafe_allow_html=True)

            # Section 1: Select r values (Two Columns)
            st.markdown("") # Small spacer
            section_label("Select model parameter (r) values:") # Style updated
            
            selected_r_indices = []
            
            # Create 2 columns for compact layout
            r_cols = st.columns(2, gap="small", vertical_alignment="center")
            
            for i, r in enumerate(r_values):
                # Place in left (0) or right (1) column based on index
                with r_cols[i % 2]:
                    if st.checkbox(f"r = {r:.3f}", value=(i == 0), key=f"r_check_{i}"):
                        selected_r_indices.append(i)

            # Section 2: Select model bias values (Two Columns)
            st.markdown("") # Small spacer
            section_label("Select model bias (Δr) values:") # Style updated
            
            selected_mb_indices = []
            
            # Create 2 columns for compact layout
            mb_cols = st.columns(2, gap="small", vertical_alignment="center")
            
            for j, mb in enumerate(model_bias_values):
                with mb_cols[j % 2]:
                    if st.checkbox(f"Δr = {mb:.1e}", value=(j == 0), key=f"mb_check_{j}"):
                        selected_mb_indices.append(j)

            st.markdown("") # Small spacer before button

            # Create current params dict
            pred_current_params = {
                'ensemble_metric': ensemble_metric,
                'r_indices': selected_r_indices,
                'mb_indices': selected_mb_indices
            }
            
            # Reset button if params changed
            if st.session_state.get('pred_last_params') != pred_current_params:
                st.session_state.pred_button_color = 'normal'
                st.session_state.plot_pred_clicked = False
            
            # Determine button color
            pred_button_type = 'secondary' if st.session_state.pred_button_color == 'success' else 'primary'

            # Plot button
            plot_button = st.button(
                "▶️ Generate Plot",
                type=pred_button_type,
                width='stretch',
                key="pred_plot_button"
            )

            # Store selections
            st.session_state.selected_r_indices = selected_r_indices
            st.session_state.selected_mb_indices = selected_mb_indices

            if plot_button:
                st.session_state.plot_pred_clicked = True  # Make state persistent
                st.session_state.pred_last_params = pred_current_params
                st.session_state.pred_button_color = 'success'
                st.rerun()  # Force reload to immediately show the green button


    elif selected_tab == 3:
        # --- BOX 1: SYSTEM PARAMETER ---
        with st.container(border=True):
            st.markdown("### System Parameter")
            r_ref_val = st.slider(
                "Reference r", 
                min_value=3.6, max_value=4.0, value=3.7, step=0.01,
                key="sb_r_ref"
            )

        # --- BOX 2: REFERENCE SCENARIO ---
        with st.container(border=True):
            st.markdown("### Reference (Base) Scenario")
            bias_options = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            
            # Model Error First (Index 0)
            mod_bias_options = [0.0] + bias_options
            ref_mod_bias = st.select_slider(
                "Reference Model Error (Δr)", 
                options=mod_bias_options, 
                value=1e-6, 
                key="sb_ref_mod",
                format_func=lambda x: f"{x:.0e}" if x > 0 else "0.0"
            )

            # IC Error Second (Index 1)
            ref_ic_bias = st.select_slider(
                "Reference Initial Error (Δx₀)", 
                options=bias_options, 
                value=1e-6, 
                key="sb_ref_ic",
                format_func=lambda x: f"{x:.0e}"
            )

        # --- BOX 3: ADDITIONAL SCENARIOS ---
        with st.container(border=True):
            st.markdown("### Additional Scenarios")
            
            # Dropdown for Additional Samples
            num_additional = st.selectbox(
                "Number of Additional Scenarios",
                [1, 2, 3, 4, 5],
                index=2, # Default = 3 (index 2) -> Total 4 lines
                key="sb_num_samples"
            )
            
            # --- Dynamic Input Grid ---
            # Column Headers
            h1, h2 = st.columns(2)
            with h1: st.markdown("<div style='font-size:14px; font-weight:600'>Model Err. (Δr)</div>", unsafe_allow_html=True)
            with h2: st.markdown("<div style='font-size:14px; font-weight:600'>Initial Err. (Δx₀)</div>", unsafe_allow_html=True)

            scenario_inputs = []
            
            # 1. Add Reference Scenario (Always Index 0, Black) - Internally kept, not displayed
            scenario_inputs.append({
                'mod': ref_mod_bias,
                'ic': ref_ic_bias,
                'color': 'black'
            })
            
            # Default values for ADDITIONAL rows
            additional_defaults = [
                (2.5e-5, 1e-6), 
                (7.5e-5, 1e-6), 
                (7.5e-5, 5e-5),
                (1e-4, 1e-4),   
                (1e-4, 1e-4)    
            ]
            
            colors_additional = ['purple', 'green', 'orange', '#1f77b4', '#d62728']
            
            # Generate Grid for Additional Scenarios
            for i in range(num_additional):
                c1, c2 = st.columns(2)
                
                def_mod, def_ic = additional_defaults[i] if i < len(additional_defaults) else (1e-4, 1e-4)
                
                with c1:
                    s_mod = st.text_input(
                        f"add_mod_hidden_{i}", 
                        value=f"{def_mod:.1e}", 
                        key=f"scen_mod_{i}",
                        label_visibility="collapsed"
                    )
                with c2:
                    s_ic = st.text_input(
                        f"add_ic_hidden_{i}", 
                        value=f"{def_ic:.1e}", 
                        key=f"scen_ic_{i}",
                        label_visibility="collapsed"
                    )
                
                scenario_inputs.append({
                    'mod': float(s_mod), 
                    'ic': float(s_ic),
                    'color': colors_additional[i % len(colors_additional)]
                })

        # --- BOX 4: ENSEMBLE METRIC ---
        with st.container(border=True):
            st.markdown("### Ensemble Metric to Plot")
            fig4_metric = st.selectbox("Metric", ["Median", "Mean", "Mode"], index=0, key="sb_fig4_metric", label_visibility="collapsed")

        # --- BOX 5: PLOT SETTINGS ---
        with st.container(border=True):
            st.markdown("### Plot Settings")
            
            # === HELPER: RESET AXIS LIMITS ===
            def reset_fig4_limits():
                st.session_state.sb_y_limit_norm = 3 # Default 10^3 for Normalized
                st.session_state.sb_y_limit_abs = -10
                st.session_state.sb_x_limit = 60 # Default X limit

            st.markdown("#### Plot Type")
            fig4_plot_type = st.radio(
                "Plot Type", 
                ["Normalized Error", "Absolute Error"], 
                index=0, 
                key="sb_fig4_plot_type", 
                on_change=reset_fig4_limits,
                label_visibility="collapsed"
            )
            
            st.markdown("#### Plot Limits")
            col_ax1, col_ax2 = st.columns(2)
            with col_ax1:
                x_limit = st.number_input("X-Axis Limit (Steps)", 20, 500, 60, 10, key="sb_x_limit")
            with col_ax2:
                if fig4_plot_type == "Normalized Error":
                    y_limit_exp = st.number_input("Y-Axis Max (10^x)", 0, 5, 3, 1, key="sb_y_limit_norm", help="Set the upper limit exponent (e.g. 3 means 10^3 = 1000)")
                else:
                    y_limit_exp = st.number_input("Y-Axis Min (10^x)", -16, -1, -10, 1, key="sb_y_limit_abs", help="Set the lower limit exponent (e.g. -10 means 10^-10)")
            
            
        # === DYNAMIC BUTTON STATE LOGIC ===
        scen_tuple = tuple([(s['mod'], s['ic']) for s in scenario_inputs])
        
        current_fig4_params = {
            'r': r_ref_val,
            'scenarios': scen_tuple,
            'xlim': x_limit,
            #'ylim': y_limit_exp,
            'metric': fig4_metric # Added metric to params tracking
        }
        
        if 'fig4_last_params' not in st.session_state:
            st.session_state.fig4_last_params = None
        
        if st.session_state.fig4_last_params != current_fig4_params:
            st.session_state.fig4_btn_color = 'primary'
        
        btn_type = 'secondary' if st.session_state.get('fig4_btn_color') == 'success' else 'primary'

        # 5. Run Button
        if st.button("▶️ Run Comparative Analysis", type=btn_type, width='stretch', on_click=reset_fig4_limits):
            with st.spinner("Running Scenario Simulations..."):
                
                # --- SIMULATION PARAMETERS ---
                r_base = r_ref_val
                steps = x_limit 
                ens_n = 50
                thresh = 0.1
                
                # FIX 1: Run over multiple Initial Conditions (ICs)
                ic_list = np.linspace(0.2, 0.8, 10) 
                
                sim_results = []
                scenarios_to_plot = []
                
                ref_res = None
                
                for i, s in enumerate(scenario_inputs):
                    
                    # Accumulators for averaging over ICs
                    accum_stat = np.zeros(steps)
                    accum_p10 = np.zeros(steps)
                    accum_p90 = np.zeros(steps)
                    
                    # Loop over different starting positions
                    for x_start in ic_list:
                        
                        # FIX 2 & 3: Apply Bias and Correct Spread
                        res = simulator.run_simulation(
                            r_true=r_base, x0_true=x_start, 
                            r_model=r_base + s['mod'], x0_model=x_start + s['ic'],
                            num_steps=steps, pred_thresh=thresh, 
                            ensemble_enabled=True, ensemble_size=ens_n,
                            init_val_pert=s['ic'], param_pert=0.0, 
                            ensemble_stat=fig4_metric
                        )
                        
                        # Accumulate results
                        accum_stat += res['x_absdiff_stat'][:steps]
                        accum_p10 += res.get('x_absdiff_p10', np.zeros(steps))[:steps]
                        accum_p90 += res.get('x_absdiff_p90', np.zeros(steps))[:steps]

                    # Compute Average Stats across all ICs
                    avg_res = {
                        'x_absdiff_stat': accum_stat / len(ic_list),
                        'x_absdiff_p10': accum_p10 / len(ic_list),
                        'x_absdiff_p90': accum_p90 / len(ic_list)
                    }

                    if i == 0: ref_res = avg_res
                    sim_results.append(avg_res)
                    
                    # Store Plot Metadata
                    scenarios_to_plot.append({
                        'ic': s['ic'],
                        'mod': s['mod'],
                        'color': s['color'],
                        'label': f"IC={s['ic']:.1e}, Δr={s['mod']:.1e}"
                    })

                # Store Results
                st.session_state.fig4_data = {
                    'ref': ref_res,
                    'scenarios': scenarios_to_plot,
                    'results': sim_results,
                    'params': current_fig4_params,
                    'timestamp': datetime.now()
                }
                
                # === UPDATE BUTTON STATE ===
                st.session_state.fig4_ran = True
                st.session_state.fig4_last_params = current_fig4_params
                st.session_state.fig4_btn_color = 'success'
                
            st.rerun()


    elif selected_tab == 4:
        # === VERTICAL SPACER ===
        # Pushes the buttons down to align with the main content header
        st.markdown("<br>" * 12, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Information Guide")
            
            # Button 1: General Intro
            type_intro = 'primary' if st.session_state.info_sub_tab == 'intro' else 'secondary'
            if st.button("General Introduction to Chaos", type=type_intro, width='stretch', key="btn_info_intro"):
                st.session_state.info_sub_tab = 'intro'
                st.rerun()

            # Button 2: How to Use
            type_usage = 'primary' if st.session_state.info_sub_tab == 'usage' else 'secondary'
            if st.button("How to Use This Site", type=type_usage, width='stretch', key="btn_info_usage"):
                st.session_state.info_sub_tab = 'usage'
                st.rerun()

            # Button 3: About
            type_about = 'primary' if st.session_state.info_sub_tab == 'about' else 'secondary'
            if st.button("About", type=type_about, width='stretch', key="btn_info_about"):
                st.session_state.info_sub_tab = 'about'
                st.rerun()


# === MAIN CONTENT AREA ===
if selected_tab == 0:
    if st.session_state.bifurcation_computed:
        fig_bif, ax_bif = plt.subplots(figsize=get_bif_figsize(), constrained_layout=True)

        data = st.session_state.bifurcation_data
        params_used = st.session_state.get('bifurcation_params_used', {})
        
        show_density_used = params_used.get('density', True)
        colormap_used = params_used.get('colormap', 'turbo')
        use_power_scale_used = params_used.get('power_scale', True)
        gamma_used = params_used.get('gamma', 0.2)
        
        if show_density_used and 'density_matrix' in data:
            custom_cmap = create_white_based_colormap(colormap_used)
            density_data = data['density_matrix'].copy()
            density_data_nonzero = density_data[density_data > 0]
            
            if len(density_data_nonzero) > 0 and use_power_scale_used:
                vmin = np.percentile(density_data_nonzero, 1)
                vmax = np.percentile(density_data_nonzero, 99.5)
                density_masked = np.ma.masked_where(density_data <= 0, density_data)
                norm = PowerNorm(gamma=gamma_used, vmin=vmin, vmax=vmax)
                density_label = f'Density (Power Scale γ={gamma_used:.2f})'
                
                im = ax_bif.imshow(density_masked, extent=[data['r_bins'][0], data['r_bins'][-1],
                                   data['x_bins'][0], data['x_bins'][-1]], origin='lower',
                                   aspect='auto', cmap=custom_cmap, norm=norm, interpolation='bilinear')
            else:
                density_label = 'Density'
                im = ax_bif.imshow(density_data, extent=[data['r_bins'][0], data['r_bins'][-1],
                                   data['x_bins'][0], data['x_bins'][-1]], origin='lower',
                                   aspect='auto', cmap=custom_cmap, interpolation='bilinear')
            
            cbar = plt.colorbar(im, ax=ax_bif, label=density_label)
            cbar.ax.tick_params(labelsize=10)
        else:
            ax_bif.plot(data['r_array'], data['x_array'], ',', color='#32b8c6', alpha=0.5, markersize=1)
        
        r_min_used = params_used.get('r_min', 2.5)
        r_max_used = params_used.get('r_max', 4.0)
        resolution_used = params_used.get('resolution', 500)
        plot_iters_used = params_used.get('plot', 100)
        
        ax_bif.set_xlim(r_min_used, r_max_used)
        ax_bif.set_ylim(0, 1)
        ax_bif.set_xlabel('r (Model Parameter)', fontsize=13, fontweight='bold')
        ax_bif.set_ylabel('x (Time Series)', fontsize=13, fontweight='bold')
        
        title_suffix = f" (Power Scale γ={gamma_used:.2f})" if (show_density_used and use_power_scale_used) else ""
        ax_bif.set_title(f'Bifurcation Diagram{title_suffix}', fontsize=14, fontweight='bold')
        ax_bif.grid(True, alpha=0.2)
        ax_bif.text(0.02, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax_bif.transAxes,
            fontsize=8, ha='left', va='bottom', style='italic', color='gray',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Save figure with meaningful filename and base64 encoding
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        buffer = BytesIO()
        fig_bif.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<a href="data:image/png;base64,{img_base64}" download="bifurcation_diagram_{timestamp}.png">'
                    f'<img src="data:image/png;base64,{img_base64}" style="width:100%"/></a>',
                    unsafe_allow_html=True)
        plt.close()

    else:
        st.info("""**Configure parameters in the sidebar and click "▶️ Compute Bifurcation"**""")


elif selected_tab == 1:
    if st.session_state.simulation_ran and st.session_state.results is not None:
        results = st.session_state.results
        timestamp = st.session_state.get('figure_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # === DYNAMIC STATISTIC SELECTION ===
        # Select the array based on the current sidebar choice (not the old simulation run)
        # Note: We fallback to 'Mean' if the key is missing (e.g. single run)
        if use_ensemble:
            stat_key = f'ensemble_{central_stat.lower()}'
            
            # --- FALLBACK LOGIC ---
            # If the specific metric (e.g. Median) wasn't computed in the original run,
            # we must compute it NOW from the full ensemble data if available.
            if stat_key not in results and 'x_model_full' in results:
                full_data = results['x_model_full']
                if central_stat == 'Median':
                    results['ensemble_median'] = np.median(full_data, axis=0)
                elif central_stat == 'Mean':
                    results['ensemble_mean'] = np.mean(full_data, axis=0)
                # Mode is too complex to compute on-the-fly reliably without scipy.stats.mode on continuous data
                # So we skip Mode dynamic calc or show warning
            
            if stat_key in results:
                x_model_dynamic = results[stat_key]
            else:
                x_model_dynamic = results['x_model_stat'] # Last resort fallback
                if central_stat == 'Mode' and stat_key not in results:
                    st.warning("⚠️ 'Mode' statistic not available in cached data. Please click 'Run Simulation' to compute it.")
        else:
            x_model_dynamic = results['x_model_det']
            
        # === RECOMPUTE ERROR & PREDICTABILITY DYNAMICALLY ===
        # This allows instant updates without re-running the simulation
        abs_diff_dynamic = np.abs(x_model_dynamic - results['x_true'])
        
        # Find new predictability index
        exceeds_dynamic = np.where(abs_diff_dynamic > pred_thresh)[0]
        pred_idx_dynamic = exceeds_dynamic[0] if len(exceeds_dynamic) > 0 else len(results['x_true'])

        # Grid Layout for Standard Plots
        if is_mobile_layout():
            col_single, = st.columns(1)
            cols = [col_single, col_single, col_single, col_single]
        else:
            # CHANGED: First two plots stacked vertically (Full Width)
            row1, = st.columns(1)
            row2, = st.columns(1)
            
            # Remaining plots side-by-side (Half Width)
            col3, col4 = st.columns(2)
            
            # Map the plot order to the new layout
            cols = [row1, row2, col3, col4]

        col_idx = 0
        
        # === DYNAMIC RECALCULATION (Instant Update) ===
        # (Logic handled above)
        
        # CHANGED: Time array now starts at 1
        time = np.arange(1, len(results['x_true']) + 1)
        
        # --- 1. TIME SERIES (STANDARD) ---
        if show_time_series:
            with cols[col_idx]:
                # === RESTORED DEFINITION LINE ===
                # Use wider figure size (12, 4) for stacked layout, unless mobile
                figsize_ts = (12, 4) if not is_mobile_layout() else get_plot_figsize()
                
                fig_ts, ax_ts = plt.subplots(figsize=figsize_ts, constrained_layout=True)

                # === UPDATED: Plot Ensemble Range based on Switch ===
                if use_ensemble and show_ensemble_spread:
                    # Logic: Check user switch AND ensure data exists
                    if st.session_state.get('ens_spread_type', "10th-90th Percentiles") == "10th-90th Percentiles" and 'x_model_p10' in results:
                        lower_b = results['x_model_p10']
                        upper_b = results['x_model_p90']
                        lbl_fill = 'Ens. Range (10-90%)'
                    else:
                        # Fallback to Min/Max (or if user selected Min-Max)
                        lower_b = results.get('x_model_min', results['x_model_stat'])
                        upper_b = results.get('x_model_max', results['x_model_stat'])
                        lbl_fill = 'Ens. Range (Min-Max)'

                    ax_ts.fill_between(time, lower_b, upper_b,
                                      alpha=0.2, color='gray', label=lbl_fill)
                
                # Dynamic Pred Limit Line
                if 0 < pred_idx_dynamic < len(time):
                    # Shift x by +1 to match 1-based indexing
                    ax_ts.axvline(x=pred_idx_dynamic + 1, color='k', linestyle='--', linewidth=1.0,
                                label=f'Pred Limit (t={pred_idx_dynamic + 1})', alpha=0.7)
                    
                # Truth
                ax_ts.plot(time, results['x_true'], 'b-', linewidth=1.0, label='Truth', alpha=0.8)
                
                # Dynamic Model Line
                label_main = f'Model ({central_stat})' if use_ensemble else 'Model (Single)'
                ax_ts.plot(time, x_model_dynamic, color='orange', linewidth=1.0, 
                          label=label_main, alpha=0.8)
                
                ax_ts.set_xlabel('Time Step (i)', fontsize=10, fontweight='bold')
                ax_ts.set_title(f'State Time Series', fontsize=11, fontweight='bold')
                ax_ts.set_ylim([-0.05, 1.3])
                ax_ts.grid(True, alpha=0.3)
                ax_ts.legend(fontsize=9, ncol=2)
                ax_ts.text(0.99, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax_ts.transAxes,
                    fontsize=8, ha='right', va='bottom', style='italic', color='gray',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                st.pyplot(fig_ts)
                col_idx += 1
        
        # --- 2. TIME SERIES DIFF (STANDARD) ---
        if show_time_series_diff:
            with cols[col_idx]:
                # === RESTORED DEFINITION LINE ===
                # Use wider figure size (12, 4) for stacked layout
                figsize_ts = (12, 4) if not is_mobile_layout() else get_plot_figsize()
                
                fig_tsd, ax_tsd = plt.subplots(figsize=figsize_ts, constrained_layout=True)
                
                # Use dynamic diff
                abs_diff = abs_diff_dynamic 

                # === UPDATED: Plot Ensemble Range based on Switch ===
                if use_ensemble and show_ensemble_spread:
                    if st.session_state.get('ens_spread_type', "10th-90th Percentiles") == "10th-90th Percentiles" and 'x_absdiff_p10' in results:
                        lower_b = np.maximum(results['x_absdiff_p10'], 1e-16)
                        upper_b = np.maximum(results['x_absdiff_p90'], 1e-16)
                        lbl_fill = 'Ens. Range (10-90%)'
                    else:
                        lower_b = np.maximum(results.get('x_absdiff_min', 1e-16), 1e-16)
                        upper_b = np.maximum(results.get('x_absdiff_max', 1e-16), 1e-16)
                        lbl_fill = 'Ens. Range (Min-Max)'

                    ax_tsd.fill_between(time, lower_b, upper_b, alpha=0.2, color='gray', label=lbl_fill)
                
                ax_tsd.axhline(y=pred_thresh, color='b', linestyle='--', linewidth=1.0, label=' Error Threshold', alpha=0.7)
                
                # Dynamic Pred Limit
                if 0 < pred_idx_dynamic < len(time):
                    # Shift x by +1 to match 1-based indexing
                    ax_tsd.axvline(x=pred_idx_dynamic + 1, color='k', linestyle='--', linewidth=1.0, 
                                   label=f'Predictability Limit (t={pred_idx_dynamic + 1})', alpha=0.7)
                    
                # Split curve based on dynamic limit
                if pred_idx_dynamic > 0:
                    ax_tsd.semilogy(time[:pred_idx_dynamic+1], abs_diff[:pred_idx_dynamic+1], 'g-', linewidth=1.0, label='Good Predictability', alpha=0.8)
                if pred_idx_dynamic < len(time) - 1:
                    ax_tsd.semilogy(time[pred_idx_dynamic:], abs_diff[pred_idx_dynamic:], 'r-', linewidth=1.0, label='Poor Predictability', alpha=0.8)
                
                ax_tsd.set_xlabel('Time Step (i)', fontsize=10, fontweight='bold')
                ax_tsd.set_title('Absolute Error (Model-Truth)', fontsize=11, fontweight='bold')
                ax_tsd.grid(True, alpha=0.3, which='both')
                ax_tsd.legend(fontsize=9, ncol=2)
                ax_tsd.text(0.02, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax_tsd.transAxes,
                    fontsize=8, ha='left', va='bottom', style='italic', color='gray',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                st.pyplot(fig_tsd)
                col_idx += 1
        
        # --- 3. STATE-SPACE (FOLDED ATTRACTORS) ---
        if show_state_space:
            with cols[col_idx]:
                fig_ss, ax_ss = plt.subplots(figsize=get_plot_figsize(), constrained_layout=True)
                
                # Background: Attractors
                xpara = np.linspace(0, 1, 200)
                ypara_truth = xpara.copy()
                ypara_model = xpara.copy()
                for _ in range(iter_diff):
                    ypara_truth = r_true * ypara_truth * (1 - ypara_truth)
                    ypara_model = r_model * ypara_model * (1 - ypara_model)
                    
                ax_ss.plot([0, 1], [0, 1], 'k--', alpha=0.3, zorder=1)
                ax_ss.plot(xpara, ypara_truth, 'b-', alpha=0.5, linewidth=1, label=f"Truth (r={r_true:.2f})", zorder=1)
                ax_ss.plot(xpara, ypara_model, color='orange', linestyle='--', alpha=0.6, linewidth=1, label=f"Model (r={r_model:.2f})", zorder=1)

                # Use Dynamic Model Data
                x_n = x_model_dynamic[:-iter_diff]
                x_n1 = x_model_dynamic[iter_diff:]
                
                # Color main scatter by time
                colors = np.arange(len(x_n))
                scatter = ax_ss.scatter(x_n, x_n1, c=colors, cmap='turbo', s=25, alpha=0.7, 
                                      edgecolors='black', label=f'Ens. Metric: {central_stat}', zorder=2)
                
                cbar = plt.colorbar(scatter, ax=ax_ss)
                cbar.set_label('Iteration Number', fontsize=10)

                # Overlays (Strictly Solid Colors)
                if use_ensemble:
                    if st.session_state.get('viz_show_mean', False) and central_stat != 'Mean':
                        xm = results.get('ensemble_mean', results['x_model_stat'])
                        ax_ss.plot(xm[:-iter_diff], xm[iter_diff:], 'o', linestyle='None', color='orange', markersize=3, alpha=0.6, label='Ens. Traj. (Mean)', zorder=3)
                    
                    if st.session_state.get('viz_show_median', False) and central_stat != 'Median':
                        xm = results.get('ensemble_median', results['x_model_stat'])
                        ax_ss.plot(xm[:-iter_diff], xm[iter_diff:], 'o', linestyle='None', color='purple', markersize=3, alpha=0.6, label='Ens. Traj. (Median)', zorder=3)
                    
                    if st.session_state.get('viz_show_mode', False) and central_stat != 'Mode' and 'ensemble_mode' in results:
                        xmod = results['ensemble_mode']
                        ax_ss.plot(xmod[:-iter_diff], xmod[iter_diff:], 'o', linestyle='None', color='deeppink', markersize=3, alpha=0.6, label='Ens. Traj. (Mode)', zorder=3)
                        
                    if st.session_state.get('viz_show_traj_mean', False):
                        xt = results['x_traj_mean']
                        ax_ss.plot(xt[:-iter_diff], xt[iter_diff:], 'o', linestyle='None', color='green', markersize=3, alpha=0.6, label='Det. Traj. (Mean)', zorder=3)

                    if st.session_state.get('viz_show_traj_median', False):
                        xt = results['x_traj_median']
                        ax_ss.plot(xt[:-iter_diff], xt[iter_diff:], 'o', linestyle='None', color='blue', markersize=3, alpha=0.6, label='Det. Traj. (Median)', zorder=3)
                        
                    if st.session_state.get('viz_show_traj_mode', False):
                        xt = results['x_traj_mode']
                        ax_ss.plot(xt[:-iter_diff], xt[iter_diff:], 'o', linestyle='None', color='teal', markersize=3, alpha=0.6, label='Det. Traj. (Mode)', zorder=3)

                ax_ss.set_xlabel(r"$\mathbf{x_{i}}$", fontsize=12, fontweight='bold')
                ax_ss.set_ylabel(r"$\mathbf{x_{i+\Delta}}$", fontsize=12, fontweight='bold')
                ax_ss.set_title(f'Attractor Geometry (Δ={iter_diff})', fontsize=11, fontweight='bold')
                ax_ss.set_xlim([0, 1])
                ax_ss.set_ylim([0, 1])
                ax_ss.grid(True, alpha=0.3)
                ax_ss.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
                ax_ss.text(0.5, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax_ss.transAxes,
                    fontsize=8, ha='center', va='bottom', style='italic', color='gray',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                st.pyplot(fig_ss)
                col_idx += 1
        
        # --- 4. STATE-SPACE DIFF (STANDARD) ---
        if show_state_space_diff:
            with cols[col_idx]:
                fig_ssd, ax_ssd = plt.subplots(figsize=get_plot_figsize(), constrained_layout=True)
                
                x_n_true = results['x_true'][:-iter_diff]
                x_n1_true = results['x_true'][iter_diff:]
                
                # Use Dynamic Model Data
                x_n_model = x_model_dynamic[:-iter_diff]
                x_n1_model = x_model_dynamic[iter_diff:]
                
                diff_x_n = x_n_model - x_n_true
                diff_x_n1 = x_n1_model - x_n1_true
                colors = np.arange(len(diff_x_n))

                scatter = ax_ssd.scatter(diff_x_n, diff_x_n1, c=colors, cmap='turbo', s=20, alpha=0.7, edgecolors='black')
                ax_ssd.axhline(0, color='k', linestyle='--', alpha=0.3); ax_ssd.axvline(0, color='k', linestyle='--', alpha=0.3)
                ax_ssd.set_xlabel(r"$\mathbf{\Delta x_{i}}$", fontsize=12, fontweight='bold')
                ax_ssd.set_ylabel(r"$\mathbf{\Delta x_{i+\Delta}}$", fontsize=12, fontweight='bold')
                ax_ssd.set_title(f'State-Space Difference (Δ={iter_diff})', fontsize=11, fontweight='bold')
                ax_ssd.grid(True, alpha=0.3)
                ax_ssd.set_xlim([-1, 1])
                ax_ssd.set_ylim([-1, 1])
                ax_ssd.text(0.99, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax_ssd.transAxes,
                    fontsize=8, ha='right', va='bottom', style='italic', color='gray',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                st.pyplot(fig_ssd)
                col_idx += 1

        # === 5. NEW SECTION: ENSEMBLE ANALYSIS (APPENDED) ===
        if use_ensemble and 'x_model_full' in results:
            
            # --- Analysis Plot 1: Multi-Metric Time Series ---
            fig_comp, ax_comp = plt.subplots(figsize=(12, 4), constrained_layout=True)
            
            # Truth:
            ax_comp.plot(time, results['x_true'], 'k-', alpha=0.2, linewidth=2.0, label='Truth')
            
            # Dynamic Pred Limit Line (Added)
            if 0 < pred_idx_dynamic < len(time):
                # Shift x by +1 to match 1-based indexing
                ax_comp.axvline(x=pred_idx_dynamic + 1, color='k', linestyle='--', linewidth=1.0,
                            label=f'Pred Limit (t={pred_idx_dynamic + 1})', alpha=0.7)
            
            # Dynamic plotting based on checkboxes:
            if st.session_state.get('viz_show_mean', False):
                # Fallback to current stat if key missing
                dat = results.get('ensemble_mean', results['x_model_stat'] if central_stat=='Mean' else None)
                if dat is not None: ax_comp.plot(time, dat, color='orange', linewidth=1.0, label='Ens. Mean')

            if st.session_state.get('viz_show_median', False):
                dat = results.get('ensemble_median', results['x_model_stat'] if central_stat=='Median' else None)
                if dat is not None: ax_comp.plot(time, dat, color='purple', linewidth=1.0, label='Ens. Median')

            if st.session_state.get('viz_show_mode', False):
                dat = results.get('ensemble_mode', results['x_model_stat'] if central_stat=='Mode' else None)
                if dat is not None: ax_comp.plot(time, dat, color='deeppink', linewidth=1.0, label='Ens. Mode')

            if st.session_state.get('viz_show_traj_mean', False):
                ax_comp.plot(time, results['x_traj_mean'], color='green', linestyle='--', linewidth=1.0, label='Det. From Mean')
            if st.session_state.get('viz_show_traj_median', False):
                ax_comp.plot(time, results['x_traj_median'], color='blue', linestyle='--', linewidth=1.0, label='Det. From Median')
            if st.session_state.get('viz_show_traj_mode', False):
                ax_comp.plot(time, results['x_traj_mode'], color='teal', linestyle='--', linewidth=1.0, label='Det. From Mode')
            
            # FIX: Updated Y-limit to 1.3 to prevent legend overlap
            ax_comp.set_xlabel('Time Step (i)', fontsize=10, fontweight='bold')
            ax_comp.set_title(f'Ensemble Metrics vs. Deterministic Trajectories', fontsize=11, fontweight='bold')
            ax_comp.set_ylim(-0.05, 1.3)
            ax_comp.legend(loc='upper center', ncol=3, fontsize=9)
            ax_comp.grid(True, alpha=0.3)
            ax_comp.text(0.99, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax_comp.transAxes,
                fontsize=8, ha='right', va='bottom', style='italic', color='gray',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            st.pyplot(fig_comp)
            
            # --- Analysis Plot 2: Histograms ---
            from matplotlib.ticker import MaxNLocator

            # Read directly from the live slider keys
            t1 = st.session_state.get('hist_t1', 10)
            t2 = st.session_state.get('hist_t2', 30)
            t3 = st.session_state.get('hist_t3', 60)
            times = [t1, t2, t3]
            
            fig_hist, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, constrained_layout=True)
            full_ens_data = results['x_model_full'] 
            
            for i, t_val in enumerate(times):
                ax = axes[i]
                
                # CHANGED: Convert 1-based slider value to 0-based array index
                idx = t_val - 1
                
                if idx < full_ens_data.shape[1]:
                    data_t = full_ens_data[:, idx]
                    truth_val = results['x_true'][idx]
                    
                    # === 1. Calculate Dynamic Range ===
                    # Start with ensemble data AND Truth (so truth is never cut off)
                    vals_to_include = [data_t, truth_val]
                    
                    # Defensively add stats if they exist
                    if st.session_state.get('viz_show_mean', False) and 'ensemble_mean' in results:
                        vals_to_include.append(results['ensemble_mean'][idx])
                    if st.session_state.get('viz_show_median', False) and 'ensemble_median' in results:
                        vals_to_include.append(results['ensemble_median'][idx])
                    
                    # ... (rest of histogram plotting remains similar but protected keys) ...
                    
                    # Determine range
                    all_vals = np.concatenate([np.atleast_1d(v) for v in vals_to_include])
                    v_min, v_max = np.min(all_vals), np.max(all_vals)
                    
                    span = v_max - v_min
                    if span < 1e-9: span = 0.05 
                    pad = span * 0.1
                    
                    plot_min = max(0.0, v_min - pad)
                    plot_max = min(1.0, v_max + pad)
                    
                    # === 2. Plot Histogram ===
                    ax.hist(data_t, bins=30, range=(plot_min, plot_max), density=True, 
                           color='skyblue', edgecolor='white', alpha=0.7, 
                           label='Ens. Dist.' if i == 0 else None)
                    
                    # === 3. Plot KDE (Conditional) ===
                    # UPDATED: Only plot if user selected "Yes"
                    if st.session_state.get("viz_show_kde", "Yes") == "Yes":
                        try:
                            kde = gaussian_kde(data_t)
                            x_grid = np.linspace(plot_min, plot_max, 200)
                            ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='KDE' if i == 0 else None)
                        except: pass
                    
                    # === 4. Plot TRUTH (Black Line) ===
                    ax.axvline(truth_val, color='black', linewidth=1.0, linestyle='-', 
                              alpha=0.8, label='Truth' if i == 0 else None, zorder=5)

                    # === 5. Plot Metric Overlays (USING idx) ===
                    if st.session_state.get('viz_show_mean', False) and 'ensemble_mean' in results:
                        ax.axvline(results['ensemble_mean'][idx], color='orange', linewidth=1.0, label='Ens. Mean' if i==0 else "")
                    
                    # ... (abbreviated, applied same safe access logic) ...

                    ax.set_xlabel('State Value Range', fontsize=10, fontweight='bold')    
                    ax.set_title(f"Ens. Histogram at i={t_val}", fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(plot_min, plot_max)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

                    ax.text(0.95, 0.95, f'© {datetime.now().year} Altug Aksoy', transform=ax.transAxes,
                       fontsize=8, ha='right', va='top', style='italic', color='gray',
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                    
                    if i == 0:
                        ax.legend(fontsize=8, loc='upper left', frameon=True, 
                                 facecolor='white', framealpha=0.9, ncol=2)

            st.pyplot(fig_hist)

        elif use_ensemble and 'x_model_full' not in results:
            st.warning("⚠️ Ensemble simulation enabled. Please click **'▶️ Run Simulation'** to generate the ensemble data.")

    else:
        st.info("""**Configure parameters in the sidebar and click "▶️ Run Simulation"**""")


elif selected_tab == 2:
    # === HANDLE GENERATE PLOT BUTTON ===
    if st.session_state.get('plot_pred_clicked', False):
        # Get selections and metric from sidebar
        selected_r_indices = st.session_state.get('selected_r_indices', [0])
        selected_mb_indices = st.session_state.get('selected_mb_indices', [0])
        y_min, y_max = st.session_state.get('pred_y_range', (0, 120))
        metric = st.session_state.get('pred_ensemble_metric', 'median')
        
        # Get data
        pred_data = st.session_state.pred_data
        r_vals = pred_data['r_values']
        ic_vals = pred_data['ic_bias_values']
        dr_vals = pred_data['model_bias_values']
        
        # Select surface based on chosen metric
        surf = pred_data['surface'][metric] if isinstance(pred_data['surface'], dict) else pred_data['surface']
        
        if selected_r_indices and selected_mb_indices:
            # Create single plot with all selected combinations
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define colors for different Δr values (map to number of Δr choices)
            num_dr = len(selected_mb_indices)
            if num_dr == 1:
                colors = ['#1f77b4']  # Blue
            elif num_dr == 2:
                colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange
            elif num_dr == 3:
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            elif num_dr == 4:
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
            elif num_dr == 5:
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Add Purple
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, num_dr))
            
            # Define line styles for different r values
            line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # Solid, dashed, dash-dot, dotted, custom
            
            # Plot all selected (r, Δr) combinations on same graph (lines only, no markers)
            for j_dr, i_dr in enumerate(selected_mb_indices):
                for k_r, i_r in enumerate(selected_r_indices):
                    r_val = r_vals[i_r]
                    dr_val = dr_vals[i_dr]
                    pred_limits = surf[i_r, i_dr, :]
                    
                    # Reverse pred_limits to match reversed x-axis (1e-3 to 1e-13)
                    pred_limits_reversed = pred_limits[::-1]
                    
                    linestyle = line_styles[k_r % len(line_styles)]
                    color = colors[j_dr] if isinstance(colors, list) else colors[j_dr]
                    
                    ax.plot(range(len(ic_vals)), pred_limits_reversed, linewidth=2.5,
                           color=color, linestyle=linestyle)
            
            ax.set_xlabel('Initial Condition Uncertainty', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predictability Limit', fontsize=12, fontweight='bold')
            ax.set_title(f'Predictability Limits (Ensemble Metric: {metric.capitalize()})',
                        fontsize=14, fontweight='bold')
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            
            # Label x-axis with IC values (reversed: 1e-3 to 1e-13)
            ax.set_xticks(range(0, len(ic_vals), max(1, len(ic_vals)//10)))
            tick_indices = ax.get_xticks().astype(int)
            ax.set_xticklabels([f'{ic_vals[len(ic_vals)-1-i]:.1e}' if i < len(ic_vals) else ''
                               for i in tick_indices], rotation=45, ha='right')
            
            # === CREATE TWO SIDE-BY-SIDE LEGENDS ===
            # Left legend: r values with line styles (gray color)
            left_legend_handles = []
            for k_r, i_r in enumerate(selected_r_indices):
                r_val = r_vals[i_r]
                linestyle = line_styles[k_r % len(line_styles)]
                left_legend_handles.append(
                    plt.Line2D([0], [0], color='gray', linewidth=2.5, linestyle=linestyle, 
                              label=f'r = {r_val:.2f}')
                )
            
            # Right legend: Δr values with colors (solid line)
            right_legend_handles = []
            for j_dr, i_dr in enumerate(selected_mb_indices):
                dr_val = dr_vals[i_dr]
                color = colors[j_dr] if isinstance(colors, list) else colors[j_dr]
                right_legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=2.5, linestyle='-', 
                              label=f'Δr = {dr_val:.1e}')
                )
            
            # Place legends side by side in upper left
            left_legend = ax.legend(handles=left_legend_handles, loc='upper left', 
                                   fontsize=10, title='r Values', title_fontsize=10,
                                   frameon=True, fancybox=True, shadow=False)
            left_legend.get_title().set_fontweight('bold')
            ax.add_artist(left_legend)
            
            right_legend = ax.legend(handles=right_legend_handles, loc='upper left', 
                                    fontsize=10, title='Δr Values', title_fontsize=10,
                                    frameon=True, fancybox=True, shadow=False,
                                    bbox_to_anchor=(0.125, 1.0))
            right_legend.get_title().set_fontweight('bold')
            
            # Add copyright text
            ax.text(0.99, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom', style='italic', color='gray',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            
            # Save figure with meaningful filename and base64 encoding
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            st.markdown(f'<a href="data:image/png;base64,{img_base64}" download="predictability_limit_{timestamp}.png">'
                       f'<img src="data:image/png;base64,{img_base64}" style="width:100%"/></a>',
                       unsafe_allow_html=True)
            plt.close(fig)

            # Mark computation as complete (turns run button green)
            st.session_state.button_color = 'success'

            st.markdown("---")
            
            # === RESULTS TABLE (only shown after plot is generated) ===
            st.markdown("### Results Table")
            st.markdown("Predictability limit (iterations) for selected parameter combinations")
            
            # Build table for selected combinations only
            table_data = []
            for i_r in selected_r_indices:
                for i_dr in selected_mb_indices:
                    r_val = r_vals[i_r]
                    dr_val = dr_vals[i_dr]
                    pred_limits_original = surf[i_r, i_dr, :]
                    
                    row = {
                        'r': f'{r_val:.2f}',
                        'Δr': f'{dr_val:.1e}',
                    }
                    
                    # Add IC bias columns in reversed order (reverse the predictability values too)
                    pred_limits_reversed = pred_limits_original[::-1]
                    for display_idx in range(len(ic_vals)):
                        ic_val = ic_vals[len(ic_vals) - 1 - display_idx]
                        col_name = f'Δx₀={ic_val:.1e}'
                        row[col_name] = int(pred_limits_reversed[display_idx])
                    
                    table_data.append(row)
            
            df_display = pd.DataFrame(table_data)
            st.dataframe(df_display, width='stretch', hide_index=True)
            
            # === DOWNLOAD CSV ===
            csv_string = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_string,
                file_name=f"predictability_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Please select at least one r/Δr combination in the sidebar.")
    else:
        st.info("""**Configure parameters in the sidebar and click "▶️ Generate Plot"**""")


elif selected_tab == 3:
    if st.session_state.get('fig4_ran', False) and 'fig4_data' in st.session_state:
        data = st.session_state.fig4_data
        
        # Helper vars
        plot_type = st.session_state.get("sb_fig4_plot_type", "Normalized Error")
        r_used = data['params']['r']
        # UPDATED: Retrieve the metric used in the calculation
        metric_used = data['params'].get('metric', 'Median')
        
        # Retrieve user-defined limits
        user_xlim = data['params'].get('xlim', 60)
        
        # Read Y-limit directly from session state
        if plot_type == "Normalized Error":
            user_ylim_exp = st.session_state.get("sb_y_limit_norm", 3)
        else:
            user_ylim_exp = st.session_state.get("sb_y_limit_abs", -10)
        
        # Create Plot
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        steps = user_xlim 
        time_axis = np.arange(1, steps + 1)
        thresh = 0.1
        
        # Get Base Reference Metric
        ref_curve_full = data['ref']['x_absdiff_stat']
        ref_curve = np.maximum(ref_curve_full[:steps], 1e-16)
        
        # --- PLOT LOGIC ---
        if plot_type == "Normalized Error":
            # 1. Plot Normalized Threshold
            norm_thresh_curve = thresh / ref_curve
            ax4.plot(time_axis, norm_thresh_curve, 'k--', linewidth=1.5, label='Error Threshold', alpha=0.8)
            
            # 2. Plot Scenarios
            for i, s in enumerate(data['scenarios']):
                res = data['results'][i]
                res_curve = res['x_absdiff_stat'][:steps] 
                norm_curve = res_curve / ref_curve
                
                base_color = s['color']
                
                # Plot Metric Line
                ax4.plot(time_axis, norm_curve, color=base_color, linewidth=2.0, alpha=0.8, 
                        label=s['label'])
                
                # Plot Spread
                if 'x_absdiff_p10' in res:
                    lower = (res['x_absdiff_p10'][:steps]) / ref_curve
                    upper = (res['x_absdiff_p90'][:steps]) / ref_curve
                    
                    ax4.plot(time_axis, lower, color=base_color, linewidth=0.5, alpha=0.4)
                    ax4.plot(time_axis, upper, color=base_color, linewidth=0.5, alpha=0.4)
            
            # UPDATED LABEL: Added back the definition (Delta / Delta_ref)
            ax4.set_ylabel(f"Normalized Error ({metric_used}, $\Delta / \Delta_{{ref}}$)", fontsize=12, fontweight='bold')
            ax4.set_title(f"Comparative Error Growth | r={r_used:.2f}", fontweight='bold')
            
            # Apply Limits: Normalized
            ax4.set_ylim(bottom=0.01, top=10**user_ylim_exp)
            
            legend_loc = 'upper right'

        else:
            # Absolute Plotting
            ax4.axhline(thresh, color='k', linestyle='--', linewidth=1.5, label='Threshold (0.1)')
            
            for i, s in enumerate(data['scenarios']):
                res = data['results'][i]
                res_curve = res['x_absdiff_stat'][:steps]
                base_color = s['color']
                
                # Plot Metric
                ax4.plot(time_axis, res_curve, color=base_color, linewidth=2.0, alpha=0.8,
                        label=s['label'])
                
                # Plot Spread
                if 'x_absdiff_p10' in res:
                    lower = res['x_absdiff_p10'][:steps]
                    upper = res['x_absdiff_p90'][:steps]
                    ax4.plot(time_axis, lower, color=base_color, linewidth=0.5, alpha=0.4)
                    ax4.plot(time_axis, upper, color=base_color, linewidth=0.5, alpha=0.4)

            # UPDATED LABEL: Dynamic based on Metric
            ax4.set_ylabel(f"Absolute Error ({metric_used})", fontsize=12, fontweight='bold')
            ax4.set_title(f"Absolute Error Growth Comparison | r={r_used:.2f}", fontweight='bold')
            
            # Apply Limits: Absolute
            ax4.set_ylim(bottom=10**user_ylim_exp, top=2.0)
            
            legend_loc = 'lower right'

        ax4.set_yscale('log')
        ax4.set_xlabel("Iteration Step", fontsize=12, fontweight='bold')
        ax4.grid(True, which="both", alpha=0.3)
        ax4.set_xlim(1, steps)

        # UPDATED: Add a dummy invisible line so the legend shows the black thin line entry
        ax4.plot([], [], color='k', linewidth=0.5, label='10th/90th Percentiles')
        
        # Legend
        ax4.legend(loc=legend_loc, fontsize=9, framealpha=0.9)

        # Add Copyright text
        if plot_type == "Normalized Error":
            cp_x, cp_ha = 0.99, 'right'
        else:
            cp_x, cp_ha = 0.02, 'left'

        ax4.text(cp_x, 0.02, f'© {datetime.now().year} Altug Aksoy', transform=ax4.transAxes,
            fontsize=8, ha=cp_ha, va='bottom', style='italic', color='gray',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        st.pyplot(fig4)

    else:
        st.info("Configure settings in the sidebar and click **▶️ Run Comparative Analysis**")


elif selected_tab == 4:
    # === SUB-TAB 1: GENERAL INTRO TO CHAOS ===
    if st.session_state.info_sub_tab == 'intro':
        st.markdown("### General Introduction to Chaos")

        st.write("""
        #### Overview
        This **Logistic Map Simulator** demonstrates how **small initial-condition errors** grow 
        exponentially in chaotic systems, eventually exceeding our ability to make accurate predictions.

        #### Key Concepts
        - **Butterfly Effect:** Tiny differences in initial conditions lead to vastly different outcomes.
        - **Predictability Limit:** Beyond this point, the model becomes useless for specific forecasts.
        - **Ensemble Forecasting:** Multiple scenarios help quantify forecast uncertainty.
        - **Ensemble-Based vs. Single-Realization Simulations:** Simulations of the selected ensemble statistics will vary from one another as well as a single/deterministic realization.
        - **Practical Applications:** This insight is critical for weather, climate, and hurricane forecasting.

        #### The Logistic Map Equation

        The simulation uses the 1D logistic map:

        $$x_{i+1} = r x_i (1 - x_i)$$

        **Parameters:**
        - $i \\geq 1$ : Iteration number of the simulation 
        - $r \\in [0, 4]$ : Model parameter that controls system behavior (Chaotic, periodic, or fixed-point)
        - $x_i \\in [0, 1]$ : System state at iteration $i$

        **Key Properties:**
        - For $r < 1$: Fixed point at $x = 0$
        - For $1 < r < 3$: Fixed point at $x = \\frac{r-1}{r}$
        - For $r > 3$: Period-doubling bifurcations lead to chaos
        - At $r \\approx 3.57$: Onset of chaos (Feigenbaum point)
        - For $r > 4$ or $r < 0$: Orbits escape to infinity
        """)

    # === SUB-TAB 2: HOW TO USE THIS SITE ===
    elif st.session_state.info_sub_tab == 'usage':
        st.markdown("### How to Use This Site")

        st.write(r"""
        #### Bifurcation
        *Visualize the long-term behavior of the system across different parameter values.*

        1.  **Set Range:** Choose the minimum and maximum $r$ values (e.g., 2.5 to 4.0) to explore different dynamical regimes.
        2.  **Resolution:** Higher resolution provides sharper images but takes longer to compute.
        3.  **Density Plot:** Enable this to see a "heatmap" of where the system spends the most time, rather than just simple points. 
        4.  **Gamma:** Adjust the Gamma slider to change the contrast of the density plot, helping reveal faint structures.

        #### Dynamics
        *Simulate the evolution of the system over time.*
        
        1.  **Set Parameters:** In the sidebar, define the **Truth** (the "real" system) and the **Model** (your simulation of it).
            * *To simulate error:* Set the initial condition $x_0$ or model $r$ to be slightly different from the Truth.
        2.  **Ensemble Simulation:** Toggle this on to simulate a "cloud" of initial conditions rather than a single point. This helps visualize uncertainty growth.
            * All ensemble simulations are generated with random seeds and will result in slightly different "average" behavior for different runs.
            * "Average" ensemble behavior can be measured by the choice of the mean, median, or mode statistic of the ensemble.
        3.  **Visuals:**
            * **Time Series:** Shows the trajectory of $x$ over "time" (i.e., iteration number $i$).
            * **Absolute Error:** Shows the difference between Model and Truth (log scale).
            * **Attractor Geometry:** A state-space plot ($x_i$ vs $x_{i+\Delta}$) revealing the shape of the chaos where $\Delta$ is the lag parameter.
            * **Histograms:** Ensemble histograms help visualize how the ensemble diverges from Gaussianity over time.

        #### Predictability
        *Analyze the limit of prediction accuracy.*

        1.  **Select Parameters:** Choose specific values of $r$ (system behavior) and $\Delta r$ (Model Bias) from the sidebar checkboxes.
        2.  **Generate Plot:** Click the button to calculate the **Predictability Limit**.
        3.  **Interpretation:**
             * The resulting plot shows how the **Predictability Limit** (the time step where error becomes too large) decreases as initial condition uncertainty increases.
             * **Without model error**, Predictability Limit can be extended **indefinitely** by reducing the initial condition uncertainty/error.
             * The introduction of **model error** demonstrates how Predictability Limit **saturates** and cannot be improved further by lowering initial condition error alone.

        #### Comparative Error Growth
        *Compare how errors grow under different model and initial condition biases.*
        
        1.  **Reference Scenario:** Define the baseline system parameter ($r$), model error ($\Delta r$), and initial condition error ($\Delta x_0$).
        2.  **Additional Scenarios:** Add up to 5 alternative scenarios with different biases to compare against the reference.
        3.  **Ensemble Metric:** Choose whether to track the Median, Mean, or Mode of the ensemble error.
        4.  **Plot Settings:**
            * **Normalized Error:** Scales all curves by the reference error, useful for seeing relative growth rates (as in Fig. 4 of the paper).
            * **Absolute Error:** Shows the raw error magnitude on a log scale.
        5.  **Interpretation:**
            * This visualization helps identify which **error source** dominates **ensemble variability**.
            * In the **presence of model error**, lower initial error may lead to situations where **ensembles collapse**.
            * **Model error, therefore, imposes an additional burden on predictability** if maintaining sufficient ensemble variability is important.
        """)

    # === SUB-TAB 3: ABOUT ===
    elif st.session_state.info_sub_tab == 'about':
        st.markdown("### About")
        
        st.markdown("#### Citation")
        st.markdown("""
        Aksoy, A., A Monte Carlo approach to understanding the impacts of initial-condition 
        uncertainty, model uncertainty, and simulation variability on the predictability of 
        chaotic systems. Perspectives from the one-dimensional logistic map. 
        *Chaos*, 34, 011102 (2024). [https://doi.org/10.1063/5.0181705](https://doi.org/10.1063/5.0181705)
        """)

        st.markdown("#### Author")
        st.write("""
        **Altug Aksoy**
        - CIMAS/Rosenstiel School, University of Miami
        - Hurricane Research Division/AOML, NOAA
        - Email: aaksoy@miami.edu | altug.aksoy@noaa.gov
        - ORCID: [0000-0002-2335-7710](https://orcid.org/0000-0002-2335-7710)
        - Webpage: [NOAA/HRD](https://www.aoml.noaa.gov/hrd/people/altugaksoy/)
        - LinkedIn: [Altug Aksoy's Profile](https://www.linkedin.com/in/altugaksoy)
        """)


st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>© 2026 Altug Aksoy | University of Miami & NOAA/AOML</p>", unsafe_allow_html=True)