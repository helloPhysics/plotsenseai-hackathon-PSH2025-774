# godmode.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import io
import time
import os
import math 

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

# ------------------------------------------------------------------------------
# --- CRITICAL: SESSION STATE INITIALIZATION (MOVED TO TOP) ---
# ------------------------------------------------------------------------------
# All session state keys MUST be initialized here to prevent KeyErrors on script rerun.
if 'app_page' not in st.session_state: # Add app_page initialization if used for routing
    st.session_state['app_page'] = 'god'
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
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "None / Linear Regression"
if 'preset_select' not in st.session_state:
    st.session_state['preset_select'] = "Manual Input"
if 'best_fit_equation' not in st.session_state:
    st.session_state['best_fit_equation'] = ""
if 'linear_r_squared' not in st.session_state:
    st.session_state['linear_r_squared'] = 0.0
if 'linear_slope_for_ai' not in st.session_state:
    st.session_state['linear_slope_for_ai'] = 0.0
if 'trigger_plot_on_load' not in st.session_state:
    st.session_state['trigger_plot_on_load'] = False
if 'run_plotsense_explainer' not in st.session_state:
    st.session_state['run_plotsense_explainer'] = False
if 'plotsense_explanation' not in st.session_state:
    st.session_state['plotsense_explanation'] = "" 
# ------------------------------------------------------------------------------
    
# --- Constants ---
PHYSICS_UNITS = [
    "None", "Time (s)", "Length (m)", "Mass (kg)", "Force (N)",
    "Velocity (m/s)", "Acceleration (m/s²)", "Voltage (V)",
    "Current (A)", "Resistance (Ω)", "Energy (J)", "Power (W)"
]

# --- Non-Linear Models Dictionary ---
NON_LINEAR_MODELS = {
    "None / Linear Regression": "linear",
    "Power Law: y = a * x^b": "power",
    "Exponential: y = a * exp(k*x)": "exponential",
    "Quadratic: y = a*x^2 + b*x + c": "quadratic",
}

# --- PRESET EXPERIMENTS LIBRARY ---
PRESET_EXPERIMENTS = {
    "Manual Input": None,
    "Simple Pendulum (T² vs L)": {
        'x_label': 'Length',
        'x_unit': 'Length (m)',
        'y_label': 'Period Squared',
        'y_unit': 'Time (s)',
        'model': 'linear',
        # T^2 = (4*pi^2/g) * L. Theoretical Gradient = 4*pi^2 / 9.81 ≈ 4.025
        'theoretical_grad': 4.025, 
        'x_data': "0.1, 0.2, 0.3, 0.4, 0.5",
        'y_data': "0.040, 0.810, 1.207, 1.636, 2.025", 
        'context': "The gradient of this graph should equal (4*pi^2) / g. Use your fitted gradient to calculate g.",
    },
    "Ohm's Law (V vs I)": {
        'x_label': 'Current',
        'x_unit': 'Current (A)',
        'y_label': 'Voltage',
        'y_unit': 'Voltage (V)',
        'model': 'linear',
        # V = I*R. Theoretical Gradient = Resistance (R)
        'theoretical_grad': 10.0, 
        'x_data': "0.1, 0.2, 0.3, 0.4, 0.5",
        'y_data': "1.02, 1.98, 3.05, 3.95, 5.01",
        'context': "The gradient of this graph represents the Resistance (R) of the component in Ohms ($\Omega$).",
    }
}
# -----------------------------------------------

# --- Non-Linear Model Functions ---
def power_law(x, a, b):
    """y = a * x^b - CORRECTED: Use absolute value of x for stability."""
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
def clear_inputs_local():
    """Clears the plot state for this mode."""
    st.session_state['fig'] = None
    st.session_state['plotted'] = False
    st.session_state['show_gradient'] = False
    st.session_state['gradient_markdown'] = ""
    st.session_state['selected_model'] = "None / Linear Regression"
    st.session_state['preset_select'] = "Manual Input" # Reset preset
    # Reset PlotSense state
    st.session_state['run_plotsense_explainer'] = False
    st.session_state['plotsense_explanation'] = ""
    st.session_state['best_fit_equation'] = ""
    st.session_state['linear_r_squared'] = 0.0
    st.session_state['linear_slope_for_ai'] = 0.0
    st.session_state['trigger_plot_on_load'] = False # Clear the trigger flag
    st.session_state['chat_history'] = [] # Clear chat history
    st.toast("Input fields and plot state cleared! 🗑️")

def set_gradient_state_local():
    """Triggers gradient calculation."""
    st.session_state['show_gradient'] = True
    st.session_state['fig'] = None # Forces a full re-render with new fit lines
    st.session_state['gradient_markdown'] = ""
    st.session_state['plotsense_explanation'] = "" # Clear explanation on new calc
    
def switch_page_local(page_name):
    """Function to switch the view/page."""
    st.session_state['app_page'] = page_name
    clear_inputs_local()

# --- PRESET LOADER ---
def load_preset_data():
    """Loads data and settings from the selected preset."""
    preset_key = st.session_state['preset_select']
    if preset_key != "Manual Input":
        data = PRESET_EXPERIMENTS[preset_key]
        
        # Reset the plot trigger flag
        st.session_state['trigger_plot_on_load'] = False 
        
        # Update input widgets via session state keys
        st.session_state[f"God_x_label"] = data['x_label']
        st.session_state[f"God_y_label"] = data['y_label']
        st.session_state[f"God_x_unit_select"] = data['x_unit']
        st.session_state[f"God_y_unit_select"] = data['y_unit']
        st.session_state[f"God_x_input"] = data['x_data']
        st.session_state[f"God_y_input"] = data['y_data']
        
        # Update curve fit model
        model_display = [k for k, v in NON_LINEAR_MODELS.items() if v == data['model']][0]
        st.session_state['selected_model'] = model_display
        
        # SET THE NEW FLAG TO TRIGGER PLOT IN run_god_mode
        st.session_state['trigger_plot_on_load'] = True
        
        st.toast(f"Preset '{preset_key}' loaded! 🧪")


# --- AI Chat Handler Functions ---

def get_chat_response(user_query, groq_key, x_label, y_label, x_unit, y_unit):
    """Generates an AI response based on the user query and current graph context."""
    global PLOTSENSE_MODEL
    
    if not groq_key:
        return "❌ Error: Groq API Key is not set. Please enter it above."

    if not st.session_state['plotted']:
         return "ℹ️ Please plot your data first to give the AI context for your questions."

    try:
        from groq import Groq # Ensure Groq is available
        client = Groq(api_key=groq_key)
        
        # Build comprehensive context from session state
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
    # Ensure the plot button key exists before accessing the chat input
    if "god_chat_input" not in st.session_state:
        return
        
    user_query = st.session_state.god_chat_input
    # Use key from the input widget, check if key is set first
    groq_key = st.session_state.get("God_groq_key") 
    
    if not user_query:
        return

    # 1. Add user query to history and clear input
    st.session_state['chat_history'].append({"role": "user", "content": user_query})
    st.session_state.god_chat_input = "" # Clear the input box

    # 2. Get AI response
    with st.spinner("AI is thinking..."):
        ai_response = get_chat_response(user_query, groq_key, x_label, y_label, x_unit, y_unit)
        
    # 3. Add AI response to history
    st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
    
    # 4. Force rerun to update chat display
    st.rerun()

# --- PlotSense Decoupled Handler ---

def handle_plotsense_explanation(fig, x_label, y_label, x_unit_label, y_unit_label, linear_slope, grad_unit_display, groq_key):
    """Handles the PlotSense AI call and updates the explanation in session state. Does not call st.pyplot()."""
    global PLOTSENSE_AVAILABLE

    if not PLOTSENSE_AVAILABLE:
        st.session_state['plotsense_explanation'] = "⚠️ PlotSense library not available. Please install it."
        return
        
    if not groq_key:
        st.session_state['plotsense_explanation'] = "❌ Groq API Key is missing. Please enter it in the sidebar."
        return

    try:
        os.environ["GROQ_API_KEY"] = groq_key
        
        # Extract unit symbols for prompt clarity
        x_unit = x_unit_label.split(' ')[-1].strip('()') if x_unit_label != "None" else ""
        y_unit = y_unit_label.split(' ')[-1].strip('()') if y_unit_label != "None" else ""
        
        # --- AI Context Generation ---
        preset_data = PRESET_EXPERIMENTS.get(st.session_state['preset_select'])
        
        # 1. Base Prompt 
        system_prompt = (
            f"You are a helpful physics teaching assistant for the Graphyscs app. "
            f"Analyze the provided matplotlib graph of {y_label} vs {x_label}. "
            f"Focus exclusively on physics concepts, units, derived quantities, and the physical relationship. "
            f"**Crucially, always explain the significance of the linear regression, its gradient, and the R-Squared values.** "
            f"Use the units ({y_unit} and {x_unit}) in your analysis. Your explanation must be highly educational and use appropriate physics terminology.\n\n"
            f"--- ANALYSIS DATA ---\n"
            f"Chosen Model: {st.session_state['selected_model']}\n"
            f"Best Fit Equation: {st.session_state['best_fit_equation']}\n"
            f"Linear R-Squared: {st.session_state['linear_r_squared']:.4f}\n"
            f"Linear Gradient: {linear_slope:.4f} ({grad_unit_display})\n"
        )
        
        # 2. Add Preset Context (Comparison)
        if preset_data and preset_data.get('theoretical_grad') is not None:
            theoretical_grad = preset_data['theoretical_grad']
            # Calculate error only if theoretical grad is not zero
            if theoretical_grad != 0:
                error_percent = abs((linear_slope - theoretical_grad) / theoretical_grad) * 100
            else:
                error_percent = 0.0 
            
            system_prompt += (
                f"Experiment Context: {st.session_state['preset_select']}\n"
                f"Theoretical Gradient Expected: {theoretical_grad:.4f}\n"
                f"**TASK: Compare the Linear Gradient ({linear_slope:.4f}) to the Theoretical Gradient ({theoretical_grad:.4f}). Calculate the percentage error ({error_percent:.2f}%) and explain the meaning of the gradient and potential sources of error.**\n"
                f"Physical Law Hint: {preset_data.get('context', 'No specific hint provided.')}\n"
            )

        os.environ["PLOTSENSE_SYSTEM_PROMPT"] = system_prompt
        # This is the AI call. It updates the global state directly.
        st.session_state['plotsense_explanation'] = ps.explainer(fig) 
        
        del os.environ["PLOTSENSE_SYSTEM_PROMPT"]

    except Exception as e:
        st.session_state['plotsense_explanation'] = f"⚠️ PlotSense Explainer failed: {e}. Check your Groq API key."
    finally:
        st.session_state['run_plotsense_explainer'] = False
        if "GROQ_API_KEY" in os.environ: del os.environ["GROQ_API_KEY"] 


def trigger_plotsense(*args):
    """Sets flag to run PlotSense Explainer and immediately executes the handler."""
    
    # Simple check for the groq key from the arguments
    if not args[-1]:
        st.error("Please enter a valid Groq API Key in the sidebar before running the AI Explainer.")
        return
        
    # Call the decoupled function directly, which will update the text but not force a plot redraw
    st.session_state['plotsense_explanation'] = "Generating explanation..."
    handle_plotsense_explanation(*args)


# --- Core Logic Functions for God Mode ---
def calculate_simple_gradient(x_np, y_np, x_unit_symbol, y_unit_symbol):
    """Calculates the gradient using the Simple Two-Point Method (First and Last Data Points)."""
    if len(x_np) < 2:
        return None, None, None, None, None, None, "Need at least two data points."

    x1, y1 = x_np[0], y_np[0]
    x2, y2 = x_np[-1], y_np[-1]

    if x2 == x1:
        return None, None, None, None, None, None, "Cannot calculate gradient: X values are identical (vertical line)."

    manual_grad = (y2 - y1) / (x2 - x1)
    c = y1 - manual_grad * x1

    if not y_unit_symbol and not x_unit_symbol: grad_unit_display = ""
    elif not x_unit_symbol: grad_unit_display = f"({y_unit_symbol})"
    elif not y_unit_symbol: grad_unit_display = f"$\\frac{{1}}{{{x_unit_symbol}}}$"
    else: grad_unit_display = f"$\\frac{{{y_unit_symbol}}}{{{x_unit_symbol}}}$"
    
    grad_info = f"""
    ### Simple Gradient Analysis (Two-Point Method)
    This is calculated using the **first data point ($P_1$)** and the **last data point ($P_2$)** entered.
    
    | Point | X-Value ($x$) | Y-Value ($y$) |
    | :--- | :--- | :--- |
    | $P_1$ (First) | ${x1:.2f}$ | ${y1:.2f}$ |
    | $P_2$ (Last) | ${x2:.2f}$ | ${y2:.2f}$ |

    **Gradient ($m$):** $$m = \\frac{{\\Delta y}}{{\\Delta x}} = \\frac{{{y2 - y1:.2f}}}{{{x2 - x1:.2f}}} = {manual_grad:.4f} {grad_unit_display}$$
    """
    return manual_grad, c, x1, y1, x2, y2, grad_info

def fit_non_linear(x_np, y_np, model_key, all_grad_info):
    """Performs the non-linear curve fitting."""
    fit_func = MODEL_FUNCTIONS[model_key]
    model_name_display = [k for k, v in NON_LINEAR_MODELS.items() if v == model_key][0]
    
    try:
        p0 = None
        if model_key == "power":
            if np.any(x_np <= 0):
                st.warning("Power Law fitting with non-positive X-values can be unstable or undefined. Using absolute X for stability.")
            p0 = [1.0, 1.0]
        elif model_key == "exponential":
            p0 = [1.0, 0.1]
        elif model_key == "quadratic":
            p0 = [1.0, 1.0, 1.0]

        popt, pcov = curve_fit(fit_func, x_np, y_np, p0=p0, maxfev=5000)
        
        residuals = y_np - fit_func(x_np, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_np - np.mean(y_np))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Store equation string globally for AI
        if model_key == "power":
            params_display = f"a = {popt[0]:.4f}, b = {popt[1]:.4f}"
            eq_display = f"$y = {popt[0]:.4f} \\cdot x^{{{popt[1]:.4f}}}$"
        elif model_key == "exponential":
            params_display = f"a = {popt[0]:.4f}, k = {popt[1]:.4f}"
            eq_display = f"$y = {popt[0]:.4f} e^{{{popt[1]:.4f}x}}$"
        elif model_key == "quadratic":
            params_display = f"a = {popt[0]:.4f}, b = {popt[1]:.4f}, c = {popt[2]:.4f}"
            eq_display = f"$y = {popt[0]:.4f}x^2 + {popt[1]:.4f}x + {popt[2]:.4f}$"
            
        st.session_state['best_fit_equation'] = eq_display

        all_grad_info += f"""
        \n\n---\n\n## Non-Linear Curve Fitting: {model_name_display}
        The **Best-Fit Curve** (green solid line) is calculated using the **Least Squares Method**.
        | Parameter | Value |
        | :--- | :--- |
        | **Equation** | {eq_display} |
        | **Coefficients** | {params_display} |
        | **R-Squared ($R^2$)** | ${r_squared:.4f}$ |
        """
        return fit_func, popt, all_grad_info

    except Exception as e:
        st.session_state['best_fit_equation'] = f"Fit Failed for {model_name_display}"
        all_grad_info += f"""
        \n\n---\n\n## Non-Linear Curve Fitting: {model_name_display}
        ⚠️ **Fit Failed** due to an optimization error or domain issue: `{type(e).__name__}`. 
        """
        return None, None, all_grad_info


def data_input_sidebar_god():
    """Sidebar for God Mode data input, now includes AI chat UI."""
    global GROQ_API_KEY_INPUT 
    mode_name = 'God'
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

        # --- PRESET LOADER ---
        st.subheader("🧪 Load Experiment Preset")
        st.selectbox(
            "Select a pre-configured experiment:",
            options=list(PRESET_EXPERIMENTS.keys()),
            key='preset_select',
            on_change=load_preset_data
        )
        st.markdown("---")
        
        st.markdown("Enter comma-separated numbers.")

        st.subheader("X-Axis")
        # Use initial values if keys don't exist yet (Streamlit handles this implicitly now)
        x_label = st.text_input("X-axis Label", value="Time", key=f"{mode_name}_x_label")
        x_unit = st.selectbox("X-axis Unit", options=PHYSICS_UNITS, index=1, key=f"{mode_name}_x_unit_select")
        x_input = st.text_input("Enter X values", value="1, 2, 3, 4, 5", key=f"{mode_name}_x_input")

        st.markdown("---")
        st.subheader("Y-Axis")
        y_label = st.text_input("Y-axis Label", value="Position", key=f"{mode_name}_y_label")
        y_unit = st.selectbox("Y-axis Unit", options=PHYSICS_UNITS, index=2, key=f"{mode_name}_y_unit_select")
        y_input = st.text_input("Enter Y values", value="1, 4, 9, 16, 25", key=f"{mode_name}_y_input")

        st.markdown("---")
        st.subheader("Curve Fitting Model")
        selected_model_display = st.selectbox(
            "Select relationship to fit:",
            options=list(NON_LINEAR_MODELS.keys()),
            key='selected_model'
        )
        selected_model_key = NON_LINEAR_MODELS[selected_model_display]

        col_p, col_c = st.columns(2)
        with col_p:
            plot_button = st.button("Plot Data", type="primary", key=f"{mode_name}_plot_button")
        with col_c:
            st.button("Clear Plot/State", on_click=clear_inputs_local, type="secondary", key=f"{mode_name}_clear_button")
        
        st.markdown("---")

        # --- NEW: AI Chat Feature UI ---
        st.subheader("GraPhycs Ai 💬")
        
        # Display chat history
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            # Only display the last few messages for space efficiency
            # FIX: st.session_state['chat_history'] is now guaranteed to exist due to early init.
            for message in st.session_state['chat_history'][-5:]:
                st.chat_message(message["role"]).write(message['content'])
        
        # Chat input field (uses the new on_change handler)
        st.text_input(
            "Ask a question about your graph or physics concept:",
            key="god_chat_input",
            # Pass current labels/units to the handler for context
            on_change=lambda: handle_ai_chat(x_label, y_label, x_unit, y_unit) 
        )
        st.caption("Press Enter to send. Requires Groq Key above.")
        # ---------------------------
        
        st.markdown("---")
        st.button("Exit to Mode Selector", on_click=lambda: switch_page_local('selector'), key=f"{mode_name}_exit_button")

    return x_input, y_input, x_label, y_label, x_unit, y_unit, plot_button, selected_model_key

def plot_god_graph(x_str, y_str, x_label, y_label, x_unit_label, y_unit_label, selected_model_key, show_gradient):
    """Parses data and creates the plot for God Mode, including regression."""
    
    # --- Helper function to clear non-data lines ---
    def clear_non_data_lines(ax):
        lines_to_remove = []
        for line in ax.lines:
            if line.get_label() and line.get_label() != 'Data Points':
                lines_to_remove.append(line)
        
        for line in lines_to_remove:
            line.remove()
        
        valid_lines = [line for line in ax.lines if line.get_label() is not None and line.get_label() != '_nolegend_']
        if valid_lines:
            ax.legend(handles=valid_lines, labels=[l.get_label() for l in valid_lines])
        elif ax.get_legend():
            ax.get_legend().remove()
    # -------------------------------------------------------------------------

    try:
        x_values = [float(val.strip()) for val in x_str.split(',') if val.strip()]
        y_values = [float(val.strip()) for val in y_str.split(',') if val.strip()]
        if not (x_values and y_values):
            st.error("Please enter data for both the X and Y axes."); return None
        if len(x_values) != len(y_values):
            st.error("The number of X values must match the number of Y values."); return None
    except ValueError:
        st.error("Invalid input. Please ensure all values are numbers."); return None

    x_np, y_np = np.array(x_values), np.array(y_values)
    
    # Create a NEW figure if the state requires a complete redraw
    if st.session_state.get('fig') is not None:
        plt.close(st.session_state['fig'])
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_np, y_np, marker='o', linestyle='', color='b', label='Data Points')

    x_unit = x_unit_label.split(' ')[-1].strip('()') if x_unit_label != "None" else ""
    y_unit = y_unit_label.split(' ')[-1].strip('()') if y_unit_label != "None" else ""
    
    # Initialize grad_unit_display
    grad_unit_display = "" 
    
    ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
    ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
    ax.set_title(f"Plot of {y_label} vs {x_label} (God Mode)")
    ax.grid(True, linestyle='--')
    all_grad_info = ""
    
    # --- ALWAYS Calculate Linear Regression for AI Critique and PlotSense Data ---
    linear_slope, linear_intercept, linear_r, _, _ = linregress(x_np, y_np)
    st.session_state['linear_r_squared'] = linear_r**2
    st.session_state['linear_slope_for_ai'] = linear_slope # Store slope for decoupled AI call

    # --- Linear Gradient Analysis ---
    if show_gradient:
        # Clear all previous regression/fit lines before drawing the new ones
        clear_non_data_lines(ax)
        
        grad, c, x1, y1, x2, y2, info = calculate_simple_gradient(x_np, y_np, x_unit, y_unit)
        if grad is not None:
            ax.plot(x_np, c + grad * x_np, 'y--', label=f'Manual Grad. Line (m: {grad:.4f})')
            ax.plot([x_np[0], x_np[-1]], [y_np[0], y_np[-1]], 's', color='y', markersize=8, label=f'Points Used: $P_1$ & $P_N$')
            all_grad_info += info
            
        # Define grad_unit_display for the linear fit and for the PlotSense block
        if not y_unit and not x_unit: grad_unit_display = ""
        elif not x_unit: grad_unit_display = f"({y_unit})"
        elif not y_unit: grad_unit_display = f"$\\frac{{1}}{{{x_unit}}}$"
        else: grad_unit_display = f"$\\frac{{{y_unit}}}{{{x_unit}}}$"
            
        # Use pre-calculated linear fit values
        ax.plot(x_np, linear_intercept + linear_slope * x_np, 'r-', label=f'Linear Regression (m: {linear_slope:.4f})')
        
        st.session_state['best_fit_equation'] = f"$y = {linear_slope:.4f}x + {linear_intercept:.4f}$"
            
        all_grad_info += f"""
        \n\n---\n\n## Linear Regression (Best-Fit Line)
        | Parameter | Value | Unit |
        | :--- | :--- | :--- |
        | **Equation** | {st.session_state['best_fit_equation']} | |
        | **Gradient ($m$)** | ${linear_slope:.4f}$ | {grad_unit_display} |
        | **Y-Intercept ($c$)** | ${linear_intercept:.4f}$ | ({y_unit}) |
        | **R-Squared ($R^2$)** | ${st.session_state['linear_r_squared']:.4f}$ | (Closer to 1 is a better linear fit) |
        """
    
    # --- Non-Linear Curve Fitting ---
    elif selected_model_key != 'linear' and not show_gradient:
        clear_non_data_lines(ax) 
        
        fit_func, popt, all_grad_info = fit_non_linear(x_np, y_np, selected_model_key, all_grad_info)
        
        if fit_func and popt is not None:
            x_fit = np.linspace(x_np.min(), x_np.max(), 500)
            y_fit = fit_func(x_fit, *popt)
            model_name_short = selected_model_key.capitalize()
            valid_indices = np.isfinite(y_fit)
            ax.plot(x_fit[valid_indices], y_fit[valid_indices], 'g-', linewidth=2, label=f'{model_name_short} Fit')
            
    else: # Only plotting initial data or linear model without gradient button
        
        # Define grad_unit_display for the PlotSense block 
        if not y_unit and not x_unit: grad_unit_display = ""
        elif not x_unit: grad_unit_display = f"({y_unit})"
        elif not y_unit: grad_unit_display = f"$\\frac{{1}}{{{x_unit}}}$"
        else: grad_unit_display = f"$\\frac{{{y_unit}}}{{{x_unit}}}$"
        
        if selected_model_key == 'linear':
            clear_non_data_lines(ax)
            ax.plot(x_np, linear_intercept + linear_slope * x_np, 'r-', label=f'Linear Regression (m: {linear_slope:.4f})')
            st.session_state['best_fit_equation'] = f"$y = {linear_slope:.4f}x + {linear_intercept:.4f}$"
        
        # If no fitting is chosen, clear previous fits
        clear_non_data_lines(ax)
        
    st.session_state['gradient_markdown'] = all_grad_info
    
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
    
    st.markdown("Using **God Mode**.")
    st.warning("This mode performs data plotting, **Non-Linear Curve Fitting**, and **Linear Analysis** (Gradient/Regression).")
    
    # 1. Get inputs from sidebar
    x_input, y_input, x_label, y_label, x_unit, y_unit, plot_button, selected_model_key = data_input_sidebar_god()
    
    # 2. Check for the button click OR the manual gradient state OR the new plot trigger flag
    if plot_button or st.session_state.get('show_gradient') or st.session_state.get('trigger_plot_on_load'):
        
        loading_placeholder = st.empty()
        with loading_placeholder.container(): st.markdown("**Thinking...** 🧠"); time.sleep(0.5)

        # plot_god_graph now returns the figure AND the calculated unit
        result = plot_god_graph(
            x_input, y_input, x_label, y_label, x_unit, y_unit, selected_model_key, st.session_state['show_gradient']
        )
        
        if result is not None:
            st.session_state['fig'], current_grad_unit = result
            st.session_state['plotted'] = True
        else:
            st.session_state['plotted'] = False

        st.session_state['show_gradient'] = False
        st.session_state['trigger_plot_on_load'] = False # Reset the trigger flag
        loading_placeholder.empty()
    
    # 3. Display plot and output
    if st.session_state['plotted'] and st.session_state['fig']:
        
        # Recalculate/retrieve the current grad unit display 
        x_unit_symbol = x_unit.split(' ')[-1].strip('()') if x_unit != "None" else ""
        y_unit_symbol = y_unit.split(' ')[-1].strip('()') if y_unit != "None" else ""
        if not y_unit_symbol and not x_unit_symbol: current_grad_unit = ""
        elif not x_unit_symbol: current_grad_unit = f"({y_unit_symbol})"
        elif not y_unit_symbol: current_grad_unit = f"$\\frac{{1}}{{{x_unit_symbol}}}$"
        else: current_grad_unit = f"$\\frac{{{y_unit_symbol}}}{{{x_unit_symbol}}}$"
        
        st.pyplot(st.session_state['fig']) # Draw the plot (this does not reset the zoom/pan on subsequent runs)
        st.markdown("---")
        
        # --- PlotSense Explanation Display and Button ---
        col_calc, col_plotsense, col_down, _ = st.columns([2, 2, 2, 2])
        
        with col_plotsense:
            if PLOTSENSE_AVAILABLE:
                st.button(
                    "Get AI Explanation (PlotSense) 🤖", 
                    on_click=trigger_plotsense, 
                    type="secondary",
                    args=(
                        st.session_state['fig'], 
                        x_label, 
                        y_label, 
                        x_unit, 
                        y_unit, 
                        st.session_state['linear_slope_for_ai'], 
                        current_grad_unit, 
                        GROQ_API_KEY_INPUT
                    )
                )
            else:
                st.info("⚠️ Install 'plotsense' for AI features.")
        
        with col_calc:
            is_linear_analysis = selected_model_key == 'linear' or st.session_state.get('selected_model') == 'None / Linear Regression'
            calc_button_label = "Calculate Linear Gradient"
            if not is_linear_analysis:
                calc_button_label = "Calculate Linear Gradient (Overrides Non-Linear Fit)"
                
            st.button(calc_button_label, on_click=set_gradient_state_local, type="primary")
        
        with col_down:
            buf = io.BytesIO()
            st.session_state['fig'].savefig(buf, format="png", bbox_inches='tight')
            st.download_button("Download Plot", buf, "godplot.png", "image/png")
        
        st.markdown("---")
        
        if st.session_state.get('plotsense_explanation'): 
            st.markdown("## PlotSense AI Interpretation 🧠 (Powered by Groq)")
            st.markdown(st.session_state['plotsense_explanation'])
            st.markdown("---")
        
        if st.session_state['gradient_markdown']:
            st.markdown(st.session_state['gradient_markdown'], unsafe_allow_html=True)
            st.markdown("---")