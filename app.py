# app.py

import streamlit as st
from basic import run_basic_mode
from godmode import run_god_mode

# --- IMAGE DISPLAY (Only here in app.py) ---
# This image will display once at the top of the entire application.
st.image("assets/GraPhycs.png", width=1000)

# --- Configuration and Initialization ---
st.set_page_config(page_title="GraPhycs3", layout="wide")

st.title("Graphycs 📊")
st.subheader("Physics Experiment Data Graph Plotter")

# Initialize global session state
if 'app_page' not in st.session_state:
    st.session_state['app_page'] = 'selector'  # Default page is the selector
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'plotted' not in st.session_state:
    st.session_state['plotted'] = False
if 'show_gradient' not in st.session_state:
    st.session_state['show_gradient'] = False
if 'gradient_markdown' not in st.session_state:
    st.session_state['gradient_markdown'] = ""

# --- Helper Functions ---
def clear_inputs():
    """Clears the plot state and resets gradient state."""
    st.session_state['fig'] = None
    st.session_state['plotted'] = False
    st.session_state['show_gradient'] = False
    st.session_state['gradient_markdown'] = ""
    st.toast("Input fields and plot state cleared! 🗑️")

def switch_page(page_name):
    """Function to switch the view/page."""
    st.session_state['app_page'] = page_name
    clear_inputs()  # Clear any previous plot state when switching mode

# --- Mode Selector (Launcher) ---
def mode_selector():
    """Displays the initial screen for mode selection."""
    st.markdown("---")
    st.markdown("## Choose Your Mode")
    st.info("Select a mode below to start plotting and analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Basic Mode (Recommended)")
        st.markdown("""
        **Features:**
        * Simple data plotting.
        * Calculates the **Manual Two-Point Gradient** using the furthest data points.
        * Shows step-by-step working and formula.
        """)
        st.button("Launch Basic Mode", on_click=lambda: switch_page('basic'), type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🔬 God Mode (Advanced)")
        st.markdown("""
    **Features:**
    * All Basic Mode features.
    * **PLUS:** Statistical **Best-Fit Linear Regression** ($\mathbf{R^2}$).
    * **Non-Linear Curve Fitting:** Exponential, Power Law, and Quadratic models.
    * **Preset Experiment Loader:** Load classic physics data (e.g., Simple Pendulum).
    * **PlotSense AI Explainer (Groq):** Provides in-depth physics analysis, compares fitted gradient to theoretical values, and calculates error.
    """)
    st.button("Launch God Mode", on_click=lambda: switch_page('god'), type="secondary", use_container_width=True)
st.markdown("---")


# --- Main App Execution ---
if st.session_state['app_page'] == 'selector':
    mode_selector()
elif st.session_state['app_page'] == 'basic':
    run_basic_mode()
elif st.session_state['app_page'] == 'god':
    run_god_mode()