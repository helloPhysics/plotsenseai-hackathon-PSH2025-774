import streamlit as st

# --- Global Placeholders for Imports (Defined here in case external files don't exist yet) ---
# NOTE: If your external files (basic.py, godmode.py) are missing or error out, 
# these simple placeholders will allow the main app to run and display the selector.
def placeholder_basic_mode():
    st.header("Basic Mode 🧪")
    st.info("This is the Basic Plotting Mode. If you see this, ensure your 'basic.py' file is present and correct.")
    st.button("Return to Selector", on_click=lambda: switch_page('selector'))

def placeholder_god_mode():
    st.header("God Mode 🔬")
    st.info("This is the Advanced Plotting Mode. If you see this, ensure your 'godmode.py' file is present and correct.")
    st.button("Return to Selector", on_click=lambda: switch_page('selector'))

try:
    # Attempt to import the real functions
    from basic import run_basic_mode
    from godmode import run_god_mode
except ImportError:
    # If import fails, use the placeholders
    st.warning("Could not import 'basic.py' or 'godmode.py'. Using placeholder modes.")
    run_basic_mode = placeholder_basic_mode
    run_god_mode = placeholder_god_mode

# ----------------------------------------------------------------------
# ⭐ CRITICAL: UNCONDITIONAL GLOBAL SESSION STATE INITIALIZATION ⭐
# All keys accessed by ALL modes MUST be initialized here to prevent KeyError.
# ----------------------------------------------------------------------

# --- Core App Flow State ---
if 'app_page' not in st.session_state:
    st.session_state['app_page'] = 'selector' 

# --- Shared Plotting State ---
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'plotted' not in st.session_state:
    st.session_state['plotted'] = False
if 'show_gradient' not in st.session_state:
    st.session_state['show_gradient'] = False
if 'gradient_markdown' not in st.session_state:
    st.session_state['gradient_markdown'] = ""
if 'last_x_np' not in st.session_state:
    st.session_state['last_x_np'] = [] # Use list/placeholder instead of np.array here
if 'last_y_np' not in st.session_state:
    st.session_state['last_y_np'] = []

# --- God Mode/Advanced Features State (FIXING THE KEYERROR) ---
if 'uploaded_data_df' not in st.session_state:
    st.session_state['uploaded_data_df'] = None
if 'uploaded_x_column' not in st.session_state:
    st.session_state['uploaded_x_column'] = None
if 'uploaded_y_column' not in st.session_state:
    st.session_state['uploaded_y_column'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [] 
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "None / Linear Regression"
if 'best_fit_equation' not in st.session_state:
    st.session_state['best_fit_equation'] = ""
if 'trigger_plot_on_load' not in st.session_state:
    st.session_state['trigger_plot_on_load'] = False
if 'display_mode' not in st.session_state:
    st.session_state['display_mode'] = 'manual' 
if 'preset_select' not in st.session_state:
    st.session_state['preset_select'] = "Manual Input"
# Add any other keys used in basic.py or godmode.py here!

# ----------------------------------------------------------------------

# --- Configuration and Initial Display ---
st.set_page_config(page_title="GraPhycs3", layout="wide")

# Use a placeholder header since I don't have the image file
st.markdown("<h1 style='text-align: center; font-size: 3em;'>GraPhycs 📊</h1>", unsafe_allow_html=True)
st.subheader("Physics Experiment Data Graph Plotter")

# --- Helper Functions ---

def clear_inputs():
    """Clears the plot state and resets gradient state."""
    # This function clears the shared state variables when switching modes.
    st.session_state['fig'] = None
    st.session_state['plotted'] = False
    st.session_state['show_gradient'] = False
    st.session_state['gradient_markdown'] = ""
    st.session_state['chat_history'] = []
    # Note: We don't clear data inputs like God_x_input here, as that is mode-specific.

def switch_page(page_name):
    """Function to switch the view/page."""
    st.session_state['app_page'] = page_name
    clear_inputs()  # Clear any previous plot state when switching mode
    st.rerun() # Forces a rerun to immediately display the new page

# ----------------------------------------------------------------------
## Mode Selector (Launcher)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
## Main App Execution Router
# ----------------------------------------------------------------------

if st.session_state['app_page'] == 'selector':
    mode_selector()
elif st.session_state['app_page'] == 'basic':
    run_basic_mode()
elif st.session_state['app_page'] == 'god':
    run_god_mode()
