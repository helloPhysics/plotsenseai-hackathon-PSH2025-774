import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import io
import time
import os
import math 
import pandas as pd 

import pandas as pd
import numpy as np
# ... (Ensure all other necessary imports like streamlit are at the top)

# ------------------------------------------------------------------------------
# ⭐ SESSION STATE INITIALIZATION MUST BE AT THE VERY TOP ⭐
# ------------------------------------------------------------------------------
if 'app_page' not in st.session_state:
    st.session_state['app_page'] = 'god'
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'plotted' not in st.session_state:
    st.session_state['plotted'] = False
if 'show_gradient' not in st.session_state:
    st.session_state['show_gradient'] = False
if 'gradient_markdown' not in st.session_state:
    st.session_state['gradient_markdown'] = ""
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "None / Linear Regression"
if 'preset_select' not in st.session_state:
    st.session_state['preset_select'] = "Manual Input"
if 'best_fit_equation' not in st.session_state:
    st.session_state['best_fit_equation'] = ""
if 'linear_r_squared' not in st.session_state:
    st.session_state['linear_r_squared'] = None # Changed from 0.0 for cleaner initial state
if 'linear_slope_for_ai' not in st.session_state:
    st.session_state['linear_slope_for_ai'] = None # Changed from 0.0 for cleaner initial state
if 'trigger_plot_on_load' not in st.session_state:
    st.session_state['trigger_plot_on_load'] = False
if 'run_plotsense_explainer' not in st.session_state:
    st.session_state['run_plotsense_explainer'] = False
if 'plotsense_explanation' not in st.session_state:
    st.session_state['plotsense_explanation'] = "" 
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [] 
if 'plotsense_recommendation' not in st.session_state:
    st.session_state['plotsense_recommendation'] = None 
if 'plotsense_recommended_fig' not in st.session_state:
    st.session_state['plotsense_recommended_fig'] = None 
if 'plotsense_recommendation_analysis' not in st.session_state:
    st.session_state['plotsense_recommendation_analysis'] = ""
if 'display_mode' not in st.session_state:
    st.session_state['display_mode'] = 'manual' 
if 'last_x_np' not in st.session_state:
    st.session_state['last_x_np'] = np.array([])
if 'last_y_np' not in st.session_state:
    st.session_state['last_y_np'] = np.array([])
if 'simple_grad_df' not in st.session_state:
    st.session_state['simple_grad_df'] = None
if 'linear_reg_df' not in st.session_state:
    st.session_state['linear_reg_df'] = None
if 'non_linear_md' not in st.session_state:
    st.session_state['non_linear_md'] = ""
if 'uploaded_data_df' not in st.session_state:
    st.session_state['uploaded_data_df'] = None
# --- NEW: Store selected column names
if 'uploaded_x_column' not in st.session_state:
    st.session_state['uploaded_x_column'] = None
if 'uploaded_y_column' not in st.session_state:
    st.session_state['uploaded_y_column'] = None


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# --- Groq and PlotSense Setup ------------------------------------------------
PLOTSENSE_MODEL = "llama-3.1-8b-instant" # Define the Groq model
GROQ_API_KEY_INPUT = None # GLOBAL: Placeholder for the key from the sidebar

try:
    # Try importing the Groq library for the chat feature
    from groq import Groq
except ImportError:
    pass # Groq library is not required for app startup, only for the chat/explainer features

try:
    import plotsense as ps
    PLOTSENSE_AVAILABLE = True
except ImportError:
    PLOTSENSE_AVAILABLE = False
# ------------------------------------------------------------------------------
    
# --- Constants ---
PHYSICS_UNITS = [
    "None", "Time (s)", "Length (m)", "Mass (kg)", "Force (N)",
    "Velocity (m/s)", "Acceleration (m/s²)", "Voltage (V)",
    "Current (A)", "Resistance (Ω)", "Energy (J)", "Power (W)",
    "Volume (m³)", "Temperature (°C)"
]

# --- Non-Linear Models Dictionary ---
NON_LINEAR_MODELS = {
    "None / Linear Regression": "linear",
    "Power Law: y = a * x^b": "power",
    "Exponential: y = a * exp(k*x)": "exponential",
    "Quadratic: y = a*x^2 + b*x + c": "quadratic",
}

# --- DATA GENERATOR FUNCTION (Noise Injection) ---
def generate_data_with_noise(base_x, base_y_formula, noise_std_dev):
    """
    Generates X data, calculates base Y, and adds random noise.
    Returns: x_np (array), y_noisy (array)
    """
    
    # 1. Convert base_x string to numpy array
    x_np = np.array([float(val.strip()) for val in base_x.split(',') if val.strip()])
    
    # 2. Calculate ideal Y data using the provided formula (lambda function)
    y_ideal = base_y_formula(x_np)
    
    # 3. Generate noise
    noise = np.random.normal(0, noise_std_dev, size=len(x_np))
    
    # 4. Add noise to ideal Y data
    y_noisy = y_ideal + noise
    
    return x_np, y_noisy


# --- PRESET EXPERIMENTS LIBRARY (Using data_generator) ---
PRESET_EXPERIMENTS = {
    "Manual Input": None,
    "Simple Pendulum (T² vs L)": {
        'x_label': 'Length', 'x_unit': 'Length (m)', 'y_label': 'Period Squared', 'y_unit': 'Time (s)',
        'model': 'linear', 'theoretical_grad': 4.025, 
        'context': "The gradient of this graph should equal (4*pi^2) / g. Use your fitted gradient to calculate g (acceleration due to gravity).",
        'data_generator': lambda: generate_data_with_noise(
            base_x="0.1, 0.2, 0.3, 0.4, 0.5", base_y_formula=lambda x: 4.025 * x, noise_std_dev=0.015 
        ),
        'base_x_label': 'L (m)', 'base_y_label': 'T (s)', 
        'original_data_formula': lambda x: np.sqrt(x), 'base_x_data_formula': lambda x: x, 
        'transformation_notes': "X-Axis: $L$ (Length); Y-Axis: $T^2$ (Period Squared). Transformation is $T^2$ vs $L$."
    },
    "Ohm's Law (V vs I)": {
        'x_label': 'Current', 'x_unit': 'Current (A)', 'y_label': 'Voltage', 'y_unit': 'Voltage (V)',
        'model': 'linear', 'theoretical_grad': 10.0, 
        'context': "The gradient of this graph represents the Resistance (R) of the component in Ohms ($\\Omega$).",
        'data_generator': lambda: generate_data_with_noise(
            base_x="0.1, 0.2, 0.3, 0.4, 0.5", base_y_formula=lambda x: 10.0 * x, noise_std_dev=0.03 
        ),
        'transformation_notes': "Plotting $V$ vs $I$ is a direct linear relationship."
    },
    "Hooke's Law (Force vs Extension) 🪝": {
        'x_label': 'Extension (e)', 'x_unit': 'Length (m)', 'y_label': 'Applied Force (F)', 'y_unit': 'Force (N)',
        'model': 'linear', 'theoretical_grad': 25.0,
        'context': "The gradient of this linear plot represents the **Spring Constant (k)** in $\\text{N/m}$.",
        'data_generator': lambda: generate_data_with_noise(
            base_x="0.01, 0.02, 0.03, 0.04, 0.05", base_y_formula=lambda x: 25.0 * x, noise_std_dev=0.005
        ),
        'transformation_notes': "Plotting $F$ vs $e$ is a direct linear relationship."
    },
    "Determination of Focal Length (1/u vs 1/v) 🔎": {
        'x_label': 'Reciprocal of Object Distance (1/u)', 'x_unit': 'Length (m)', 'y_label': 'Reciprocal of Image Distance (1/v)', 'y_unit': 'Length (m)', 
        'model': 'linear', 'theoretical_grad': -1.0, 
        'context': "The **y-intercept is equal to $\\frac{1}{f}$**, where $f$ is the focal length in meters.",
        'data_generator': lambda: generate_data_with_noise(
            base_x="10, 8.33, 6.67, 5.00, 3.33", base_y_formula=lambda x: -1.0 * x + 13.33, noise_std_dev=0.1
        ),
        'base_x_label': 'u (m)', 'base_y_label': 'v (m)',
        'base_x_data_formula': lambda x: 1.0 / x, 'original_data_formula': lambda y: 1.0 / y, 
        'transformation_notes': "Plotting $\\frac{1}{v}$ vs $\\frac{1}{u}$ is a required linearization."
    },
    "Refractive Index of Glass (sin i vs sin r) 🌈": {
        'x_label': 'Sine of Angle of Refraction (sin r)', 'x_unit': 'None', 'y_label': 'Sine of Angle of Incidence (sin i)', 'y_unit': 'None',
        'model': 'linear', 'theoretical_grad': 1.5,
        'context': "The **gradient of the line represents the refractive index (n)** of the glass block.",
        'data_generator': lambda: generate_data_with_noise(
            base_x="0.174, 0.342, 0.500, 0.643, 0.766", base_y_formula=lambda x: 1.5 * x, noise_std_dev=0.01 
        ),
        'base_x_label': '$r$ (deg)', 'base_y_label': '$i$ (deg)',
        'base_x_data_formula': lambda x: np.degrees(np.arcsin(x)), 'original_data_formula': lambda y: np.degrees(np.arcsin(y)), 
        'transformation_notes': "Plotting $\\sin i$ vs $\\sin r$ is a direct linear relationship from Snell's law."
    },
    "Specific Heat Capacity (E vs ΔT) 🔥": {
        'x_label': 'Temperature Change (ΔT)', 'x_unit': 'Temperature (°C)', 'y_label': 'Electrical Energy Supplied (E)', 'y_unit': 'Energy (J)',
        'model': 'linear', 'theoretical_grad': 500.0,
        'context': "The gradient is equal to the product of the **mass (m)** and the **specific heat capacity (c)** of the solid ($m \\cdot c$).",
        'data_generator': lambda: generate_data_with_noise(
            base_x="1.0, 2.0, 3.0, 4.0, 5.0", base_y_formula=lambda x: 500.0 * x, noise_std_dev=10.0
        ),
        'transformation_notes': "Plotting $E$ vs $\\Delta T$ is a direct linear relationship."
    },
    "Potentiometer (Internal Resistance) 🔋": {
        'x_label': 'Current (I)', 'x_unit': 'Current (A)', 'y_label': 'Terminal Potential Difference (V)', 'y_unit': 'Voltage (V)',
        'model': 'linear', 'theoretical_grad': -1.0, 
        'context': "The **y-intercept is the EMF (E)**, and the **gradient is the negative of the internal resistance ($-r$)**.",
        'data_generator': lambda: generate_data_with_noise(
            base_x="0.1, 0.2, 0.3, 0.4, 0.5", base_y_formula=lambda x: 3.0 - 1.0 * x, noise_std_dev=0.008
        ),
        'transformation_notes': "Plotting $V$ vs $I$ is a direct linear relationship."
    }
}
# -----------------------------------------------

# --- Non-Linear Model Functions ---
def power_law(x, a, b):
    """y = a * x^b"""
    return a * np.power(np.abs(x), b)

def exponential_func(x, a, k):
    """y = a * exp(k*x)"""
    return a * np.exp(k * x)

def quadratic_func(x, a, b, c):
    """y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

# Map of model key to fitting function
MODEL_FUNCTIONS = {
    "power": power_law,
    "exponential": exponential_func,
    "quadratic": quadratic_func
}

# ------------------------------------------------------------------------------


# --- Helper Functions (Local to this mode) ---
def clear_inputs_local(clear_data=True):
    """Clears the plot state for this mode, and optionally the data inputs."""
    st.session_state['fig'] = None
    st.session_state['plotted'] = False
    st.session_state['show_gradient'] = False
    st.session_state['gradient_markdown'] = ""
    st.session_state['selected_model'] = "None / Linear Regression"
    st.session_state['preset_select'] = "Manual Input" 
    st.session_state['run_plotsense_explainer'] = False
    st.session_state['plotsense_explanation'] = ""
    st.session_state['best_fit_equation'] = ""
    st.session_state['linear_r_squared'] = 0.0
    st.session_state['linear_slope_for_ai'] = 0.0
    st.session_state['trigger_plot_on_load'] = False 
    st.session_state['chat_history'] = [] 
    st.session_state['plotsense_recommended_fig'] = None 
    st.session_state['display_mode'] = 'manual' 
    st.session_state['simple_grad_df'] = None
    st.session_state['linear_reg_df'] = None
    st.session_state['non_linear_md'] = ""
    
    if clear_data:
        st.session_state['last_x_np'] = np.array([])
        st.session_state['last_y_np'] = np.array([])
        st.session_state['uploaded_data_df'] = None # Clear uploaded data
        st.session_state['uploaded_x_column'] = None
        st.session_state['uploaded_y_column'] = None
        # Clear manual input fields if they exist
        if 'God_x_input' in st.session_state: st.session_state['God_x_input'] = "1, 2, 3, 4, 5"
        if 'God_y_input' in st.session_state: st.session_state['God_y_input'] = "1, 4, 9, 16, 25"
        st.toast("Input fields and plot state cleared! 🗑️")
    else:
        st.toast("Plot state cleared! Data preserved. 🗑️")


def set_gradient_state_local():
    """Triggers gradient calculation."""
    st.session_state['show_gradient'] = True
    st.session_state['fig'] = None 
    st.session_state['plotsense_recommended_fig'] = None 
    st.session_state['gradient_markdown'] = ""
    st.session_state['plotsense_explanation'] = "" 
    st.session_state['display_mode'] = st.session_state.get('display_mode', 'manual') # Preserve mode


def switch_page_local(page_name):
    """Function to switch the view/page."""
    st.session_state['app_page'] = page_name
    clear_inputs_local()

# --- PRESET LOADER ---
def load_preset_data():
    """Loads data and settings from the selected preset, generating new data if a generator exists."""
    preset_key = st.session_state['preset_select']
    
    # 1. Clear any uploaded data first
    st.session_state['uploaded_data_df'] = None
    st.session_state['uploaded_x_column'] = None
    st.session_state['uploaded_y_column'] = None
    
    if preset_key != "Manual Input":
        data = PRESET_EXPERIMENTS[preset_key]
        
        # --- GENERATE DATA ---
        if 'data_generator' in data and callable(data['data_generator']):
            x_np, y_np = data['data_generator']()
            
            st.session_state['last_x_np'] = x_np
            st.session_state['last_y_np'] = y_np
            
            # Format back to comma-separated strings for Streamlit input fields
            x_data_str = ", ".join([f"{x:.4g}" for x in x_np])
            y_data_str = ", ".join([f"{y:.4g}" for y in y_np])
            
            # Update input widgets via session state keys
            st.session_state[f"God_x_input"] = x_data_str
            st.session_state[f"God_y_input"] = y_data_str
            st.session_state[f"God_x_label"] = data['x_label']
            st.session_state[f"God_y_label"] = data['y_label']
            st.session_state[f"God_x_unit_select"] = data['x_unit']
            st.session_state[f"God_y_unit_select"] = data['y_unit']
        
        # Update curve fit model
        model_display = [k for k, v in NON_LINEAR_MODELS.items() if v == data['model']][0]
        st.session_state['selected_model'] = model_display
        
        # SET THE NEW FLAG TO TRIGGER PLOT IN run_god_mode
        st.session_state['trigger_plot_on_load'] = True
        st.session_state['plotsense_recommended_fig'] = None 
        st.session_state['display_mode'] = 'manual' 
        st.toast(f"Preset '{preset_key}' loaded with **fresh experimental data**! 🧪")
    else:
        # Revert to default manual input strings
        st.session_state[f"God_x_input"] = "1, 2, 3, 4, 5"
        st.session_state[f"God_y_input"] = "1, 4, 9, 16, 25"
        st.session_state[f"God_x_label"] = "Time"
        st.session_state[f"God_y_label"] = "Position"
        st.session_state[f"God_x_unit_select"] = "Time (s)"
        st.session_state[f"God_y_unit_select"] = "Length (m)"
        st.session_state['trigger_plot_on_load'] = False
        st.session_state['display_mode'] = 'manual'


# --- FILE UPLOADER HANDLERS (NEW) ---

def handle_file_upload(uploaded_file):
    """Reads CSV or Excel file into a pandas DataFrame and stores it."""
    
    # 1. Reset state
    clear_inputs_local(clear_data=False) 
    st.session_state['uploaded_data_df'] = None
    st.session_state['uploaded_x_column'] = None
    st.session_state['uploaded_y_column'] = None
    st.session_state['preset_select'] = "Manual Input" # Ensure preset is reset
    
    if uploaded_file is None:
        return

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    try:
        if file_extension in ['.csv', '.txt']:
            # Try common delimiters for CSV/TXT
            df = pd.read_csv(uploaded_file, encoding='utf8', sep=None, engine='python')
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error(f"Unsupported file type: {file_extension}. Please upload a CSV, TXT, or Excel file (.xlsx, .xls).")
            return

        # 2. Clean Data: Remove rows with all NaNs and ensure columns are readable strings
        df = df.dropna(how='all')
        df.columns = [str(col).strip() for col in df.columns]
        
        if df.empty:
            st.error("The file is empty or contains no readable data after cleanup.")
            return

        # 3. Store the DataFrame and auto-select first two numeric columns (if possible)
        st.session_state['uploaded_data_df'] = df
        
        # Get all columns, not just numeric, so user can select text/date columns if they want (though they might fail plotting)
        column_options = df.columns.tolist() 
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Auto-select the first two columns, prioritizing numeric ones
        if len(numeric_options := (numeric_cols + [c for c in column_options if c not in numeric_cols])) >= 2:
            st.session_state['uploaded_x_column'] = numeric_options[0]
            st.session_state['uploaded_y_column'] = numeric_options[1]
            st.toast(f"File loaded! Auto-selected '{numeric_options[0]}' and '{numeric_options[1]}'.")
            st.session_state['trigger_plot_on_load'] = True # Trigger a plot attempt
        elif len(column_options) == 1:
            st.session_state['uploaded_x_column'] = column_options[0]
            st.session_state['uploaded_y_column'] = column_options[0] # Fallback
            st.toast("File loaded! Please manually select X and Y columns.")
        else:
            st.toast("File loaded! Please manually select X and Y columns.")
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state['uploaded_data_df'] = None
        
def update_uploaded_data():
    """Forces the application to process the uploaded data and update the plot inputs."""
    
    df = st.session_state['uploaded_data_df']
    x_col = st.session_state['uploaded_x_column']
    y_col = st.session_state['uploaded_y_column']

    if df is None or x_col is None or y_col is None:
        st.warning("Please select both X and Y columns.")
        return

    try:
        # 1. Extract columns and convert to numeric (errors='coerce' turns non-numeric to NaN)
        # We try to use the selected columns regardless of their original Dtype
        x_np_full = pd.to_numeric(df[x_col], errors='coerce').values
        y_np_full = pd.to_numeric(df[y_col], errors='coerce').values
        
        # 2. Filter out NaN values from both arrays simultaneously
        valid_indices = np.isfinite(x_np_full) & np.isfinite(y_np_full)
        x_np = x_np_full[valid_indices]
        y_np = y_np_full[valid_indices]

        if len(x_np) < 2:
            st.warning(f"Selected columns contain fewer than two valid numeric data points after cleaning ({len(x_np)} found).")
            st.session_state['last_x_np'] = np.array([])
            st.session_state['last_y_np'] = np.array([])
            st.session_state['trigger_plot_on_load'] = False
            return
            
        # 3. Update the session state variables that hold the actual data for plotting
        st.session_state['last_x_np'] = x_np
        st.session_state['last_y_np'] = y_np
        
        # 4. Update the X/Y label inputs (but not the string inputs, which are hidden for file mode)
        st.session_state[f"God_x_label"] = x_col
        st.session_state[f"God_y_label"] = y_col
        
        st.session_state['trigger_plot_on_load'] = True
        st.session_state['display_mode'] = 'file'
        st.toast(f"Plotting {len(x_np)} points from file data.")


    except Exception as e:
        st.error(f"Error processing selected columns: {e}. Check if columns exist or have valid numeric data.")
        st.session_state['last_x_np'] = np.array([])
        st.session_state['last_y_np'] = np.array([])
        st.session_state['trigger_plot_on_load'] = False


# --- AI Chat Handler Functions (Unchanged) ---
def get_chat_response(user_query, groq_key, x_label, y_label, x_unit, y_unit):
    """Generates an AI response based on the user query and current graph context."""
    global PLOTSENSE_MODEL
    
    if not groq_key:
        return "❌ Error: Groq API Key is not set. Please enter it above."

    if not st.session_state['plotted']:
          return "ℹ️ Please plot your data first to give the AI context for your questions."

    try:
        from groq import Groq 
        client = Groq(api_key=groq_key)
        
        context = (
            f"You are a concise, helpful physics tutor. Answer the user's question based on the provided graph context. "
            f"Graph Context: Y-Axis is '{y_label}' ({y_unit}), X-Axis is '{x_label}' ({x_unit}).\n"
            f"Fitted Model: {st.session_state['selected_model']}\n"
            f"Best Fit Equation: {st.session_state['best_fit_equation']}\n"
            f"Linear R-Squared ($R^2$): {st.session_state['linear_r_squared']:.4f}\n"
            f"Linear Gradient: {st.session_state['linear_slope_for_ai']:.4f}\n"
            f"Last Conversation: {[f'{msg["role"]}: {msg["content"][:50]}...' for msg in st.session_state['chat_history'][-2:]]}\n"
        )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": context}] + st.session_state['chat_history'] + 
                     [{"role": "user", "content": user_query}],
            model=PLOTSENSE_MODEL,
        )
        
        return chat_completion.choices[0].message.content

    except ImportError:
        return "⚠️ Error: The 'groq' library is not installed. Run `pip install groq`."
    except Exception as e:
        return f"❌ Groq API Call Failed: {e}. Check your key or API limits."


def handle_ai_chat(x_label, y_label, x_unit, y_unit):
    """Processes the user's chat message and updates history."""
    if "god_chat_input" not in st.session_state:
        return
        
    user_query = st.session_state.god_chat_input
    groq_key = st.session_state.get("God_groq_key") 
    
    if not user_query:
        return

    st.session_state['chat_history'].append({"role": "user", "content": user_query})
    st.session_state.god_chat_input = "" 

    with st.spinner("AI is thinking..."):
        ai_response = get_chat_response(user_query, groq_key, x_label, y_label, x_unit, y_unit)
        
    st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
    st.rerun()

# --- PlotSense Decoupled Handler (Unchanged) ---
def handle_plotsense_explanation(fig, x_label, y_label, x_unit_label, y_unit_label, linear_slope, grad_unit_display, groq_key):
    """Handles the PlotSense AI call and updates the explanation in session state."""
    global PLOTSENSE_AVAILABLE

    if not PLOTSENSE_AVAILABLE:
        st.session_state['plotsense_explanation'] = "⚠️ PlotSense library not available. Please install it."
        return
        
    if not groq_key:
        st.session_state['plotsense_explanation'] = "❌ Groq API Key is missing. Please enter it in the sidebar."
        return

    try:
        os.environ["GROQ_API_KEY"] = groq_key
        
        x_unit = x_unit_label.split(' ')[-1].strip('()') if x_unit_label != "None" else ""
        y_unit = y_unit_label.split(' ')[-1].strip('()') if y_unit_label != "None" else ""
        
        preset_data = PRESET_EXPERIMENTS.get(st.session_state['preset_select'])
        
        system_prompt = (
            f"You are an expert physics teaching assistant for the Graphyscs app. "
            f"Analyze the provided matplotlib graph of '{y_label}' (Y-axis) vs '{x_label}' (X-axis). "
            f"The Y-axis quantity has base units of **{y_unit}** and the X-axis quantity has base units of **{x_unit}**. "
            f"Your explanation must be highly educational, rigorous, and use appropriate physics terminology and $\\LaTeX$ for equations. "
            f"**REQUIRED ANALYSIS POINTS:**\n"
            f"1. **Physical Law:** State the fundamental physics principle or law that governs the relationship (e.g., Ohm's Law). Write the theoretical governing equation (e.g., $V=IR$).\n"
            f"2. **Linearization:** Explain the physical significance of plotting $Y$ vs $X$. If the true relationship is non-linear, explain why this plot is used.\n"
            f"3. **Gradient's Significance:** Crucially, state the **physical quantity** the linear regression gradient represents and derive its expected **SI Units** from $\\frac{{\\text{{Unit of Y}}}}{{\\text{{Unit of X}}}} = \\frac{{{y_unit}}}{{{x_unit}}} = {grad_unit_display}$.\n"
            f"4. **Fit Quality:** Explain the meaning of the Linear R-Squared value, $R^2 = {st.session_state['linear_r_squared']:.4f}$, in terms of experimental error and correlation.\n"
            f"5. **Equation:** Reference the Best Fit Equation: {st.session_state['best_fit_equation']} and what the intercept represents physically.\n\n"
            f"--- FITTED DATA CONTEXT ---\n"
            f"Chosen Model: {st.session_state['selected_model']}\n"
            f"Linear Gradient Measured: {linear_slope:.4f} {grad_unit_display}\n"
        )
        
        if preset_data:
            if preset_data.get('theoretical_grad') is not None:
                theoretical_grad = preset_data['theoretical_grad']
                if theoretical_grad != 0:
                    error_percent = abs((linear_slope - theoretical_grad) / theoretical_grad) * 100
                else:
                    error_percent = 0.0 
                
                system_prompt += (
                    f"\n\n**6. Error Analysis (CRITICAL):** The theoretical value for the gradient is {theoretical_grad:.4f}. "
                    f"Your measured gradient ({linear_slope:.4f}) has a percentage error of **{error_percent:.2f}%**. "
                    f"Discuss the meaning of this difference and suggest at least two specific, common **physical** (not mathematical) sources of experimental error for this type of experiment. "
                    f"Physical Law Hint: {preset_data.get('context', 'No specific hint provided.')}\n"
                )

        os.environ["PLOTSENSE_SYSTEM_PROMPT"] = system_prompt
        st.session_state['plotsense_explanation'] = ps.explainer(fig) 
        
        del os.environ["PLOTSENSE_SYSTEM_PROMPT"]

    except Exception as e:
        st.session_state['plotsense_explanation'] = f"⚠️ PlotSense Explainer failed: {e}. Check your Groq API key or plotsense installation."
    finally:
        st.session_state['run_plotsense_explainer'] = False
        if "GROQ_API_KEY" in os.environ: del os.environ["GROQ_API_KEY"] 

def trigger_plotsense(*args):
    """Sets flag to run PlotSense Explainer and immediately executes the handler."""
    
    if not args[-1]:
        st.error("Please enter a valid Groq API Key in the sidebar before running the AI Explainer.")
        return
        
    st.session_state['plotsense_explanation'] = "Generating explanation..."
    handle_plotsense_explanation(*args)

def get_data_arrays(x_str, y_str):
    """Helper to parse data inputs and handle errors (ONLY for Manual/Preset data)."""
    
    if st.session_state['uploaded_data_df'] is not None and st.session_state['display_mode'] == 'file':
        # If we are in file mode, the data is already in st.session_state['last_x_np']
        if st.session_state['last_x_np'].size > 0:
            return st.session_state['last_x_np'], st.session_state['last_y_np']
        else:
            # Safety fallback: re-run data extraction for file mode if data is missing
            update_uploaded_data()
            if st.session_state['last_x_np'].size > 0:
                 return st.session_state['last_x_np'], st.session_state['last_y_np']
            else:
                 # Last resort failure
                 st.error("Uploaded file data is not ready. Please select columns and click 'Update Plot from File Data'.")
                 return None, None


    # Handle Manual/Preset Input
    try:
        x_values = [float(val.strip()) for val in x_str.split(',') if val.strip()]
        y_values = [float(val.strip()) for val in y_str.split(',') if val.strip()]
        if not (x_values and y_values):
            st.error("Please enter data for both the X and Y axes."); return None, None
        if len(x_values) != len(y_values):
            st.error("The number of X values must match the number of Y values."); return None, None
        
        x_np = np.array(x_values)
        y_np = np.array(y_values)
        
        st.session_state['last_x_np'] = x_np
        st.session_state['last_y_np'] = y_np
        
        return x_np, y_np
    except ValueError:
        st.error("Invalid input. Please ensure all values are numbers and separated by commas."); return None, None
        
# ------------------------------------------------------------------------------

# --- Function to generate data as a Streamlit table/dataframe ---
def generate_data_table_df(x_np, y_np):
    """Generates a pandas DataFrame for st.dataframe() based on data type."""
    
    if st.session_state['display_mode'] == 'file':
        # --- UPLOADED FILE MODE ---
        x_col_name = st.session_state.get('uploaded_x_column', 'X-Value')
        y_col_name = st.session_state.get('uploaded_y_column', 'Y-Value')

        # Create a DataFrame for display from the processed NumPy arrays
        df_display = pd.DataFrame({
            'No.': np.arange(1, len(x_np) + 1),
            x_col_name: [f"{x:.4g}" for x in x_np],
            y_col_name: [f"{y:.4g}" for y in y_np]
        }).set_index('No.')
        
        st.markdown("### Data Table (From Uploaded File)")
        st.caption(f"Showing **{len(x_np)}** valid, numeric rows extracted from the uploaded file.")
        
    else:
        # --- MANUAL/PRESET MODE ---
        preset_key = st.session_state['preset_select']
        data = PRESET_EXPERIMENTS.get(preset_key)
        
        data_for_df = []
        
        if data and data != "Manual Input" and data is not None and 'base_x_label' in data:
            # Transformed data case
            base_x_data_formula = data.get('base_x_data_formula', lambda x: x)
            x_original = base_x_data_formula(x_np)
            original_data_formula = data.get('original_data_formula', lambda y: y)
            y_original = original_data_formula(y_np)
            
            for i in range(len(x_np)):
                data_for_df.append({
                    'No.': i + 1,
                    data['base_x_label']: f"{x_original[i]:.4g}",
                    data['base_y_label']: f"{y_original[i]:.4g}",
                    data['x_label']: f"{x_np[i]:.4g}",
                    data['y_label']: f"{y_np[i]:.4g}"
                })
            df_display = pd.DataFrame(data_for_df).set_index('No.')
            
            st.markdown("### Data Table (Raw & Transformed)")
            st.caption(f"The table below shows the original raw measurements and the transformed values used in the plot. {data['transformation_notes']}")
                
        else:
            # Direct linear plot or Manual Input Case
            x_label_man = st.session_state.get('God_x_label', 'X-Value')
            y_label_man = st.session_state.get('God_y_label', 'Y-Value')

            for i in range(len(x_np)):
                 data_for_df.append({
                    'No.': i + 1,
                    x_label_man: f"{x_np[i]:.4g}",
                    y_label_man: f"{y_np[i]:.4g}"
                })
            
            df_display = pd.DataFrame(data_for_df).set_index('No.')
            st.markdown("### Data Table")
            st.caption("The plot uses the data entered directly, $Y$ vs $X$.")
    
    # st.dataframe provides a clean, native table view similar to Excel/Word
    st.dataframe(df_display, use_container_width=True) 
    st.markdown("---")
# ------------------------------------------------------------------------------


# --- Core Logic Functions for God Mode ---
def calculate_simple_gradient(x_np, y_np, x_unit_symbol, y_unit_symbol):
    """
    Calculates the gradient and returns the necessary dataframes/markdown for analysis.
    """
    if len(x_np) < 2:
        return None, None, None, None, None, None, None, "Need at least two data points."

    # Use the first and last points for the 'simple' gradient
    x1, y1 = x_np[0], y_np[0]
    x2, y2 = x_np[-1], y_np[-1]

    if x2 == x1:
        return None, None, None, None, None, None, None, "Cannot calculate gradient: X values are identical (vertical line)."

    manual_grad = (y2 - y1) / (x2 - x1)
    c = y1 - manual_grad * x1

    if not y_unit_symbol and not x_unit_symbol: grad_unit_display = ""
    elif not x_unit_symbol: grad_unit_display = f"({y_unit_symbol})"
    elif not y_unit_symbol: grad_unit_display = f"$\\frac{{1}}{{{x_unit_symbol}}}$"
    else: grad_unit_display = f"$\\frac{{{y_unit_symbol}}}{{{x_unit_symbol}}}$"

    # --- GRADIENT ANALYSIS DATAFRAME (Two-Point) ---
    data = {
        'Point': ['**$P_1$ (First)**', '**$P_2$ (Last)**', '**Difference**', f'**Gradient ($m$) {grad_unit_display}**'],
        'X-Value ($x$)': [f'**{x1:.4g}**', f'**{x2:.4g}**', f'**{x2 - x1:.4g}**', ''],
        'Y-Value ($y$)': [f'**{y1:.4g}**', f'**{y2:.4g}**', f'**{y2 - y1:.4g}**', f'**{manual_grad:.4f}**'],
    }
    simple_grad_df = pd.DataFrame(data).set_index('Point')
    
    simple_grad_eq = f"$y = {manual_grad:.4f}x + {c:.4f}$ (Two-Point Estimate)" 
    
    return manual_grad, c, x1, y1, x2, y2, simple_grad_df, simple_grad_eq

def fit_non_linear(x_np, y_np, model_key):
    """Performs the non-linear curve fitting and generates the markdown."""
    fit_func = MODEL_FUNCTIONS[model_key]
    model_name_display = [k for k, v in NON_LINEAR_MODELS.items() if v == model_key][0]
    
    try:
        p0 = None
        if model_key == "power":
            # Power law requires positive x values for standard log-linearization,
            # but curve_fit handles it if we pass initial guess
            if np.any(x_np <= 0):
                st.warning("Power Law fitting with non-positive X-values can be unstable or undefined. Using absolute X for stability.")
            p0 = [1.0, 1.0]
        elif model_key == "exponential":
            p0 = [1.0, 0.1]
        elif model_key == "quadratic":
            p0 = [1.0, 1.0, 1.0]

        x_data_for_fit = np.abs(x_np) if model_key == "power" and np.any(x_np <= 0) else x_np
        
        popt, pcov = curve_fit(fit_func, x_data_for_fit, y_np, p0=p0, maxfev=5000)
        
        residuals = y_np - fit_func(x_data_for_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_np - np.mean(y_np))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Generate equation string and DataFrame
        if model_key == "power":
            eq_display = f"$y = {popt[0]:.4f} \\cdot x^{{{popt[1]:.4f}}}$"
            params_df = pd.DataFrame({
                'Parameter': ['**a** (Constant)', '**b** (Exponent)', '**R-Squared ($R^2$)**'],
                'Value': [
                    f'**{popt[0]:.4f}**', 
                    f'**{popt[1]:.4f}**', 
                    f'**{r_squared:.4f}** (Closer to 1 is better fit)'
                ]
            }).set_index('Parameter')
            
        elif model_key == "exponential":
            eq_display = f"$y = {popt[0]:.4f} e^{{{popt[1]:.4f}x}}$"
            params_df = pd.DataFrame({
                'Parameter': ['**a** (Amplitude)', '**k** (Rate Constant)', '**R-Squared ($R^2$)**'],
                'Value': [
                    f'**{popt[0]:.4f}**', 
                    f'**{popt[1]:.4f}**', 
                    f'**{r_squared:.4f}** (Closer to 1 is better fit)'
                ]
            }).set_index('Parameter')
            
        elif model_key == "quadratic":
            eq_display = f"$y = {popt[0]:.4f}x^2 + {popt[1]:.4f}x + {popt[2]:.4f}$"
            params_df = pd.DataFrame({
                'Parameter': ['**a** (Quadratic Coeff.)', '**b** (Linear Coeff.)', '**c** (Y-Intercept)', '**R-Squared ($R^2$)**'],
                'Value': [
                    f'**{popt[0]:.4f}**', 
                    f'**{popt[1]:.4f}**', 
                    f'**{popt[2]:.4f}**', 
                    f'**{r_squared:.4f}** (Closer to 1 is better fit)'
                ]
            }).set_index('Parameter')

        st.session_state['best_fit_equation'] = eq_display

        non_linear_md = f"""
        \n\n---\n\n## Non-Linear Curve Fitting: {model_name_display}
        The **Best-Fit Curve** (green solid line) is calculated using the **Least Squares Method**.
        The fitted equation is: **{eq_display}**
        """
        
        return fit_func, popt, params_df, non_linear_md

    except Exception as e:
        st.session_state['best_fit_equation'] = f"Fit Failed for {model_name_display}"
        non_linear_md = f"""
        \n\n---\n\n## Non-Linear Curve Fitting: {model_name_display}
        ⚠️ **Fit Failed** due to an optimization error or domain issue: `{type(e).__name__}`. 
        The plot will only show raw data.
        """
        return None, None, None, non_linear_md


def data_input_sidebar_god():
    """Sidebar for God Mode data input, now includes File Uploader."""
    global GROQ_API_KEY_INPUT 
    mode_name = 'God'
    x_input, y_input, x_label, y_label, x_unit, y_unit = "", "", "", "", "None", "None"
    
    with st.sidebar:
        st.header(f"Data Input ({mode_name} Mode)")
        
        # --- Groq API Key Input Field ---
        if PLOTSENSE_AVAILABLE:
            st.markdown("---")
            st.subheader("🤖 AI Configuration (Groq)")
            GROQ_API_KEY_INPUT = st.text_input(
                "Groq API Key (Llama/Mixtral)",
                value="", 
                type="password", 
                key=f"{mode_name}_groq_key"
            )
            st.caption(f"Model: `{PLOTSENSE_MODEL}`. Get key from [Groq Console](https://console.groq.com/keys).")
            st.markdown("---")

        # --- FILE UPLOADER (NEW) ---
        st.subheader("📁 Upload Data File")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel (XLSX/XLS)",
            type=["csv", "xlsx", "xls", "txt"],
            on_change=lambda: handle_file_upload(st.session_state.get('uploaded_file_input')),
            key='uploaded_file_input'
        )

        # If a file is uploaded and processed, show column selection
        if st.session_state['uploaded_data_df'] is not None:
            df = st.session_state['uploaded_data_df']
            column_options = df.columns.tolist()
            st.caption(f"File: **{uploaded_file.name}** loaded with {len(column_options)} columns.")
            
            # Use column names for labels by default
            x_col = st.selectbox(
                "Select X Column",
                options=column_options,
                key='uploaded_x_column',
                index=column_options.index(st.session_state.get('uploaded_x_column', column_options[0])) if st.session_state.get('uploaded_x_column') in column_options else 0
            )
            y_col = st.selectbox(
                "Select Y Column",
                options=column_options,
                key='uploaded_y_column',
                index=column_options.index(st.session_state.get('uploaded_y_column', column_options[1] if len(column_options) > 1 else column_options[0])) if st.session_state.get('uploaded_y_column') in column_options else (1 if len(column_options) > 1 else 0)
            )
            
            # Update the labels immediately based on selection (for graph title)
            x_label = x_col
            y_label = y_col
            
            # Data strings are now dummy/ignored
            x_input = "file_data"
            y_input = "file_data"
            
            plot_button = st.button("Update Plot from File Data", on_click=update_uploaded_data, type="secondary")
            
            # Set mode to file if data is present
            if st.session_state['uploaded_data_df'] is not None:
                st.session_state['display_mode'] = 'file'

            st.markdown("---")
            
        # --- MANUAL/PRESET LOADER ---
        
        # We only show the preset/manual entry block if no file is currently loaded
        if st.session_state['uploaded_data_df'] is None:
            st.session_state['display_mode'] = 'manual'
            
            st.subheader("OR Load Preset / Manual Entry")
            st.selectbox(
                "Select a pre-configured experiment:",
                options=list(PRESET_EXPERIMENTS.keys()),
                key='preset_select',
                on_change=load_preset_data
            )
            st.caption("Selecting a preset generates new data each time.")
            st.markdown("Enter comma-separated numbers.")

            st.subheader("X-Axis")
            x_label = st.text_input("X-axis Label", value="Time", key=f"{mode_name}_x_label")
            x_unit = st.selectbox("X-axis Unit", options=PHYSICS_UNITS, index=PHYSICS_UNITS.index(st.session_state.get(f"{mode_name}_x_unit_select", "Time (s)")) if st.session_state.get(f"{mode_name}_x_unit_select") in PHYSICS_UNITS else 1, key=f"{mode_name}_x_unit_select")
            x_input = st.text_input("Enter X values", value="1, 2, 3, 4, 5", key=f"{mode_name}_x_input")

            st.markdown("---")
            st.subheader("Y-Axis")
            y_label = st.text_input("Y-axis Label", value="Position", key=f"{mode_name}_y_label")
            y_unit = st.selectbox("Y-axis Unit", options=PHYSICS_UNITS, index=PHYSICS_UNITS.index(st.session_state.get(f"{mode_name}_y_unit_select", "Length (m)")) if st.session_state.get(f"{mode_name}_y_unit_select") in PHYSICS_UNITS else 2, key=f"{mode_name}_y_unit_select")
            y_input = st.text_input("Enter Y values", value="1, 4, 9, 16, 25", key=f"{mode_name}_y_input")
            
            plot_button = st.button("Plot Data", type="primary", key=f"{mode_name}_plot_button")

        # --- Unit Selection (Shared Block) ---
        else:
            # If a file is loaded, only show the unit selectors below the data selectors
            st.subheader("Units (Optional)")
            # Use fixed key names for unit selectors regardless of data source
            x_unit = st.selectbox("X-axis Unit", options=PHYSICS_UNITS, index=PHYSICS_UNITS.index(st.session_state.get(f"{mode_name}_x_unit_select", "None")) if st.session_state.get(f"{mode_name}_x_unit_select") in PHYSICS_UNITS else 0, key=f"{mode_name}_x_unit_select")
            y_unit = st.selectbox("Y-axis Unit", options=PHYSICS_UNITS, index=PHYSICS_UNITS.index(st.session_state.get(f"{mode_name}_y_unit_select", "None")) if st.session_state.get(f"{mode_name}_y_unit_select") in PHYSICS_UNITS else 0, key=f"{mode_name}_y_unit_select")

        # --- Shared Fitting and Control ---
        st.markdown("---")
        st.subheader("Curve Fitting Model")
        selected_model_display = st.selectbox(
            "Select relationship to fit:",
            options=list(NON_LINEAR_MODELS.keys()),
            key='selected_model'
        )
        selected_model_key = NON_LINEAR_MODELS[selected_model_display]

        col_p, col_c = st.columns([1, 1])
        
        with col_p:
            # Only show plot button if not in file mode (file mode uses 'Update Plot')
            if st.session_state['uploaded_data_df'] is None:
                 pass # Plot button is defined above for manual mode
            else:
                 pass # Plot button for file mode is defined above as 'Update Plot'
        
        with col_c:
            # Modified clear button to clear all data
            st.button("Clear All", on_click=lambda: clear_inputs_local(clear_data=True), type="secondary", key=f"{mode_name}_clear_button")
        
        st.markdown("---")

        # --- AI Chat Feature UI ---
        st.subheader("GraPhycs Ai 💬")
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for message in st.session_state['chat_history'][-5:]:
                st.chat_message(message["role"]).write(message['content'])
        
        st.text_input(
            "Ask a question about your graph or physics concept:",
            key="god_chat_input",
            on_change=lambda: handle_ai_chat(x_label, y_label, x_unit, y_unit) 
        )
        st.caption("Press Enter to send. Requires Groq Key above.")
        st.markdown("---")
        st.button("Exit to Mode Selector", on_click=lambda: switch_page_local('selector'), key=f"{mode_name}_exit_button")

    # In file mode, the plot button state is handled differently. We must return a single boolean.
    plot_trigger = plot_button or st.session_state.get('trigger_plot_on_load')
    if st.session_state['uploaded_data_df'] is not None and st.session_state['uploaded_x_column'] is not None and st.session_state['uploaded_y_column'] is not None:
        st.session_state['display_mode'] = 'file'
        x_input = "file_data"
        y_input = "file_data"
    
    return x_input, y_input, x_label, y_label, x_unit, y_unit, plot_trigger, selected_model_key

def plot_god_graph(x_str, y_str, x_label, y_label, x_unit_label, y_unit_label, selected_model_key, show_gradient):
    """Parses data and creates the plot for God Mode, including regression."""
    
    def clear_non_data_lines(ax):
        lines_to_remove = [line for line in ax.lines if line.get_label() and line.get_label() != 'Data Points']
        for line in lines_to_remove: line.remove()
        valid_lines = [line for line in ax.lines if line.get_label() is not None and line.get_label() != '_nolegend_']
        if valid_lines:
            ax.legend(handles=valid_lines, labels=[l.get_label() for l in valid_lines])
        elif ax.get_legend():
            ax.get_legend().remove()
    # -------------------------------------------------------------------------

    # Get data arrays (Handles manual, preset, and file modes internally)
    x_np, y_np = get_data_arrays(x_str, y_str)
    if x_np is None or x_np.size < 2: 
        st.session_state['simple_grad_df'] = None
        st.session_state['linear_reg_df'] = None
        st.session_state['non_linear_md'] = ""
        st.session_state['plotted'] = False
        return None, None

    if st.session_state.get('fig') is not None:
        plt.close(st.session_state['fig'])
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_np, y_np, marker='o', linestyle='', color='b', label='Data Points')

    # Get unit symbols
    x_unit = x_unit_label.split(' ')[-1].strip('()') if x_unit_label != "None" else ""
    y_unit = y_unit_label.split(' ')[-1].strip('()') if y_unit_label != "None" else ""
    
    ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
    ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
    ax.set_title(f"Plot of {y_label} vs {x_label} (GraPhycs Analysis)")
    ax.grid(True, linestyle='--')

    
    # --- Linear Regression ---
    linear_slope, linear_intercept, linear_r, _, _ = linregress(x_np, y_np)
    st.session_state['linear_r_squared'] = linear_r**2
    st.session_state['linear_slope_for_ai'] = linear_slope 

    # Define grad_unit_display for the linear fit and for the PlotSense block
    if not y_unit and not x_unit: grad_unit_display = ""
    elif not x_unit: grad_unit_display = f"({y_unit})"
    elif not y_unit: grad_unit_display = f"$\\frac{{1}}{{{x_unit}}}$"
    else: grad_unit_display = f"$\\frac{{{y_unit}}}{{{x_unit}}}$"
    
    st.session_state['best_fit_equation'] = f"$y = {linear_slope:.4f}x + {linear_intercept:.4f}$"
        
    # --- Generate Linear Regression DataFrame ---
    linear_reg_data = {
        'Parameter': [
            '**Equation**', 
            f'**Gradient ($m$) {grad_unit_display}**', 
            f'**Y-Intercept ($c$) ({y_unit})**', 
            '**R-Squared ($R^2$)**'
        ],
        'Value': [
            st.session_state['best_fit_equation'], 
            f'**{linear_slope:.4f}**', 
            f'**{linear_intercept:.4f}**', 
            f'**{st.session_state["linear_r_squared"]:.4f}** (Closer to 1 is better fit)'
        ]
    }
    st.session_state['linear_reg_df'] = pd.DataFrame(linear_reg_data).set_index('Parameter')
    st.session_state['simple_grad_df'] = None
    st.session_state['non_linear_md'] = ""
    st.session_state['non_linear_df'] = None
    
    # --- Show Analysis: Linear Gradient (Two-Point) and Regression ---
    if show_gradient:
        clear_non_data_lines(ax)
        
        grad, c, x1, y1, x2, y2, simple_grad_df, simple_grad_eq = calculate_simple_gradient(
            st.session_state['last_x_np'], st.session_state['last_y_np'], x_unit, y_unit
        )
        
        st.session_state['simple_grad_df'] = simple_grad_df
        
        if grad is not None:
            ax.plot(x_np, c + grad * x_np, 'y--', label=f'Manual Grad. Line (m: {grad:.4f})')
            ax.plot([x_np[0], x_np[-1]], [y_np[0], y_np[-1]], 's', color='y', markersize=8, label=f'Points Used: $P_1$ & $P_N$')
            
        ax.plot(x_np, linear_intercept + linear_slope * x_np, 'r-', label=f'Linear Regression (m: {linear_slope:.4f})')
        
    
    # --- Show Analysis: Non-Linear Curve Fitting ---
    elif selected_model_key != 'linear' and not show_gradient:
        clear_non_data_lines(ax) 
        
        fit_func, popt, params_df, non_linear_md = fit_non_linear(x_np, y_np, selected_model_key)
        st.session_state['non_linear_md'] = non_linear_md
        st.session_state['non_linear_df'] = params_df
        
        if fit_func and popt is not None:
            x_min = x_np.min()
            x_max = x_np.max()
            # Ensure the fit line covers the range, handle log scale issues
            if selected_model_key == "power" and x_min <= 0:
                # For power law with non-positive x, we must plot only positive data or handle carefully
                x_fit = np.linspace(max(0.01, x_min), x_max, 500)
                x_data_for_fit = x_fit 
            else:
                x_fit = np.linspace(x_min, x_max, 500)
                x_data_for_fit = x_fit
            
            y_fit = fit_func(x_data_for_fit, *popt)
            model_name_short = selected_model_key.capitalize()
            valid_indices = np.isfinite(y_fit)
            ax.plot(x_fit[valid_indices], y_fit[valid_indices], 'g-', linewidth=2, label=f'{model_name_short} Fit')
            
    else: # Only plotting initial data or linear model without gradient button
        if selected_model_key == 'linear':
            clear_non_data_lines(ax)
            ax.plot(x_np, linear_intercept + linear_slope * x_np, 'r-', label=f'Linear Regression (m: {linear_slope:.4f})')
        
        clear_non_data_lines(ax)
            
    # Update legend one final time
    valid_lines = [line for line in ax.lines if line.get_label() is not None and line.get_label() != '_nolegend_']
    if valid_lines:
        ax.legend(handles=valid_lines, labels=[l.get_label() for l in valid_lines])
    elif ax.get_legend():
        ax.get_legend().remove()
        
    return fig, grad_unit_display

# --- Main Run Function ---
def run_god_mode():
    """Main function to run the God Mode UI and logic."""
    global GROQ_API_KEY_INPUT 
    
    st.markdown("# ⚙️ GraPhycs God Mode")
    st.warning("This tool performs data plotting, **Non-Linear Curve Fitting**, and **Linear Analysis** based on manual entry or **uploaded data**.")
    
    # 1. Get inputs from sidebar
    x_input, y_input, x_label, y_label, x_unit, y_unit, plot_trigger, selected_model_key = data_input_sidebar_god()
    
    # 2. Define Plot Placeholder
    plot_placeholder = st.empty() 
    
    # 3. Check for the plot trigger
    should_run_plot_logic = plot_trigger
    
    if should_run_plot_logic:
        
        loading_placeholder = plot_placeholder.container()
        with loading_placeholder: st.markdown("**Thinking...** 🧠"); time.sleep(0.5)
        
        # plot_god_graph returns the Matplotlib figure
        result = plot_god_graph(
            x_input, y_input, x_label, y_label, x_unit, y_unit, selected_model_key, st.session_state['show_gradient']
        )
        
        if result is not None:
            st.session_state['fig'], current_grad_unit = result
            st.session_state['plotted'] = True
        else:
            st.session_state['plotted'] = False

        st.session_state['show_gradient'] = False
        st.session_state['trigger_plot_on_load'] = False 
        loading_placeholder.empty()
    
    # 4. Display Plot and Output
    display_fig = st.session_state.get('fig')

    if display_fig:
        st.session_state['plotted'] = True
        
        # --- Display the Plot in the Placeholder ---
        with plot_placeholder.container():
            st.pyplot(display_fig) 
            st.markdown("---")
            
            # --- Display Data Table ---
            if st.session_state['last_x_np'].size > 0:
                generate_data_table_df(st.session_state['last_x_np'], st.session_state['last_y_np'])
                
            # Recalculate/retrieve the current grad unit display (needed for AI)
            x_unit_symbol = x_unit.split(' ')[-1].strip('()') if x_unit != "None" else ""
            y_unit_symbol = y_unit.split(' ')[-1].strip('()') if y_unit != "None" else ""
            if not y_unit_symbol and not x_unit_symbol: current_grad_unit = ""
            elif not x_unit_symbol: current_grad_unit = f"({y_unit_symbol})"
            elif not y_unit_symbol: current_grad_unit = f"$\\frac{{1}}{{{x_unit_symbol}}}$"
            else: current_grad_unit = f"$\\frac{{{y_unit_symbol}}}{{{x_unit_symbol}}}$"
            
            # --- Action Buttons ---
            col_calc, col_plotsense, col_down = st.columns([1.5, 1.5, 1])
            
            with col_calc:
                is_linear_analysis = selected_model_key == 'linear' or st.session_state.get('simple_grad_df') is not None
                st.button(
                    "Show Linear/Two-Point Analysis", 
                    on_click=set_gradient_state_local, 
                    type="primary" if is_linear_analysis else "secondary", 
                    key='show_grad_button'
                )
            
            with col_plotsense:
                if PLOTSENSE_AVAILABLE and display_fig:
                    st.button(
                        "Get AI Explanation (PlotSense) 🤖", 
                        on_click=trigger_plotsense, 
                        type="secondary",
                        args=(
                            display_fig, x_label, y_label, x_unit, y_unit, 
                            st.session_state['linear_slope_for_ai'], current_grad_unit, 
                            st.session_state.get("God_groq_key")
                        )
                    )
            
            with col_down:
                buf = io.BytesIO()
                display_fig.savefig(buf, format="png")
                st.download_button(
                    label="Download Plot ⬇️",
                    data=buf.getvalue(),
                    file_name=f"{y_label}_vs_{x_label}_godmode.png",
                    mime="image/png"
                )
            
            st.markdown("---")
            
            # --- Analysis Results Display ---
            st.markdown("## Curve Fit & Analysis Results")
            
            if st.session_state['simple_grad_df'] is not None:
                st.markdown("### 1. Simple Gradient Analysis (Two-Point Method)")
                st.caption("Calculated using the first and last data points only.")
                st.table(st.session_state['simple_grad_df'])
                st.markdown("---")

            if st.session_state['linear_reg_df'] is not None:
                st.markdown("### 2. Linear Regression (Best-Fit Line)")
                st.caption("Calculated using the Least Squares Method for all data points.")
                st.table(st.session_state['linear_reg_df'])
                st.markdown("---")
                
            if st.session_state['non_linear_md']:
                st.markdown(st.session_state['non_linear_md']) 
                if st.session_state.get('non_linear_df') is not None:
                     st.table(st.session_state['non_linear_df'])
                st.markdown("---")
                
            if st.session_state['plotsense_explanation']:
                st.markdown("## 🤖 PlotSense AI Physics Explainer")
                st.info(st.session_state['plotsense_explanation'])
            
# ------------------------------------------------------------------------------

# --- Streamlit Page Execution ---
if __name__ == "__main__":
    st.set_page_config(
        page_title="Graphycs God Mode", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    if st.session_state.get('app_page', 'god') == 'god':
        run_god_mode()
    else:
        st.write("Exited God Mode.")