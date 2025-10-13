import streamlit as st
from basic import run_basic_mode
from godmode import run_god_mode

# --- UNCONDITIONAL GLOBAL SESSION STATE INITIALIZATION ---
# This block is consolidated and placed at the very top.
# It ensures all necessary keys are set before any functions or imported code runs.

# Default page is 'selector' to show the mode choice screen
if 'app_page' not in st.session_state:
    st.session_state['app_page'] = 'selector' 

# Initialize keys used by the main app or the modes
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [] 
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'plotted' not in st.session_state:
    st.session_state['plotted'] = False
if 'show_gradient' not in st.session_state:
    st.session_state['show_gradient'] = False
if 'gradient_markdown' not in st.session_state:
    st.session_state['gradient_markdown'] = ""
# Add any other global keys here (e.g., from basic.py or godmode.py)

# --- Configuration and Initial Display ---
st.set_page_config(page_title="GraPhycs3", layout="wide")

# This image will display once at the top of the entire application.
st.image("assets/GraPhycs.png", width=1000)

st.title("Graphycs 📊")
st.subheader("Physics Experiment Data Graph Plotter")

# --- Helper Functions ---

def clear_inputs():
    """Clears the plot state and resets gradient state."""
    # Ensure this only clears generic/shared state
    st.session_state['fig'] = None
    st.session_state['plotted'] = False
    st.session_state['show_gradient'] = False
    st.session_state['gradient_markdown'] = ""
    # st.toast("Input fields and plot state cleared! 🗑️") # Removed toast to avoid running on every switch

def switch_page(page_name):
    """Function to switch the view/page."""
    # This is the critical function that changes the view
    st.session_state['app_page'] = page_name
    clear_inputs()  # Clear any previous plot state when switching mode
    st.rerun() # Forces a rerun to immediately display the new page

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
        # Ensure the on_click uses the switch_page function
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
        # Ensure the on_click uses the switch_page function
        st.button("Launch God Mode", on_click=lambda: switch_page('god'), type="secondary", use_container_width=True)
    st.markdown("---")


# --- Main App Execution Router ---
# This block reads the session state and calls the appropriate function.

if st.session_state['app_page'] == 'selector':
    mode_selector()
elif st.session_state['app_page'] == 'basic':
    run_basic_mode()
elif st.session_state['app_page'] == 'god':
    run_god_mode()