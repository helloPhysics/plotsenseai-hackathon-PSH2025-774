import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time

# --- Constants ---
PHYSICS_UNITS = [
    "None", "Time (s)", "Length (m)", "Mass (kg)", "Force (N)",
    "Velocity (m/s)", "Acceleration (m/s²)", "Voltage (V)",
    "Current (A)", "Resistance (Ω)", "Energy (J)", "Power (W)"
]

# --- Helper Functions (Local to this mode) ---
# NOTE: These functions rely on session state keys initialized in app.py.

def clear_inputs_local():
    """Clears the plot state for this mode."""
    # Only clear the state keys relevant to the plot visualization/output
    st.session_state['fig'] = None
    st.session_state['plotted'] = False
    st.session_state['show_gradient'] = False
    st.session_state['gradient_markdown'] = ""
    st.toast("Plot state cleared!")

def set_gradient_state_local():
    """Triggers gradient calculation and re-plot."""
    # By setting this flag, the main loop in run_basic_mode will re-run the plot function
    st.session_state['show_gradient'] = True
    # We clear the fig to force the plot logic to regenerate the figure immediately
    st.session_state['fig'] = None

def switch_page(page_name):
    """Function to switch the view/page back to the selector."""
    # Directly changing the app_page state will trigger a rerun of app.py
    st.session_state['app_page'] = page_name
    # Note: We rely on the main app's switch_page to call a global clear_inputs

# --- Core Logic Functions for Basic Mode ---
def calculate_manual_gradient(x_np, y_np, x_unit_symbol, y_unit_symbol):
    """Calculates the gradient using the Two-Point Method (Min/Max X)."""
    if len(x_np) < 2:
        return None, None, None, None, None, None, "Need at least two data points."

    # Use argmin/argmax on x_np to find the furthest points horizontally
    x_min_idx, x_max_idx = np.argmin(x_np), np.argmax(x_np)
    x1, y1 = x_np[x_min_idx], y_np[x_min_idx]
    x2, y2 = x_np[x_max_idx], y_np[x_max_idx]

    if x2 == x1:
        return None, None, None, None, None, None, "Cannot calculate gradient: X values are identical (vertical line)."

    manual_grad = (y2 - y1) / (x2 - x1)
    c = y1 - manual_grad * x1

    # Format the unit display using LaTeX/Markdown for clarity
    if not y_unit_symbol and not x_unit_symbol: grad_unit_display = ""
    elif not x_unit_symbol: grad_unit_display = f"({y_unit_symbol})"
    elif not y_unit_symbol: grad_unit_display = f"$\\frac{{1}}{{{x_unit_symbol}}}$"
    else: grad_unit_display = f"$\\frac{{{y_unit_symbol}}}{{{x_unit_symbol}}}$"

    grad_info = f"""
    ### Manual Gradient Analysis (Two-Point Method)
    The gradient is calculated manually using the **Two-Point Method**, assuming the line passes through the minimum and maximum X-values, $P_1$ and $P_2$.
    
    | Point | X-Value ($x$) | Y-Value ($y$) |
    | :--- | :--- | :--- |
    | $P_1$ (Min X) | ${x1:.2f}$ | ${y1:.2f}$ |
    | $P_2$ (Max X) | ${x2:.2f}$ | ${y2:.2f}$ |

    **Formula:** $$m = \\frac{{y_2 - y_1}}{{x_2 - x_1}}$$ 
    **Calculation:** $$m = \\frac{{{y2:.2f} - {y1:.2f}}}{{{x2:.2f} - {x1:.2f}}} = \\frac{{{y2 - y1:.2f}}}{{{x2 - x1:.2f}}}$$
    
    **Final Gradient ($m$)**: **${manual_grad:.4f}$** {grad_unit_display}
    **Y-Intercept ($c$)**: ${c:.4f}$ ({y_unit_symbol})
    """
    return manual_grad, c, x1, y1, x2, y2, grad_info

def data_input_sidebar_basic():
    """Sidebar for Basic Mode data input."""
    mode_name = 'Basic'
    with st.sidebar:
        st.header(f"Data Input ({mode_name} Mode)")
        st.markdown("Enter comma-separated numbers.")

        st.subheader("X-Axis")
        x_label = st.text_input("X-axis Label", value="Time", key=f"{mode_name}_x_label")
        # Find index for 'Time (s)' (index 1) as default
        x_unit = st.selectbox("X-axis Unit", options=PHYSICS_UNITS, index=1, key=f"{mode_name}_x_unit_select")
        x_input = st.text_input("Enter X values", value="1, 2, 3, 4, 5", key=f"{mode_name}_x_input")

        st.markdown("---")
        st.subheader("Y-Axis")
        y_label = st.text_input("Y-axis Label", value="Position", key=f"{mode_name}_y_label")
        # Find index for 'Length (m)' (index 2) as default
        y_unit = st.selectbox("Y-axis Unit", options=PHYSICS_UNITS, index=2, key=f"{mode_name}_y_unit_select")
        y_input = st.text_input("Enter Y values", value="2, 4.1, 5.9, 8.2, 10", key=f"{mode_name}_y_input")

        st.markdown("---")
        col_p, col_c = st.columns(2)
        with col_p:
            plot_button = st.button("Plot Data", type="primary", key=f"{mode_name}_plot_button")
        with col_c:
            st.button("Clear Plot/State", on_click=clear_inputs_local, type="secondary", key=f"{mode_name}_clear_button")

        st.markdown("---")
        # Use the non-local switch_page function
        st.button("Exit to Mode Selector", on_click=lambda: switch_page('selector'), key=f"{mode_name}_exit_button")

    # Return the raw string inputs and labels
    return x_input, y_input, x_label, y_label, x_unit, y_unit, plot_button

def plot_basic_graph(x_str, y_str, x_label, y_label, x_unit_label, y_unit_label, show_gradient):
    """Parses data and creates the plot for Basic Mode."""
    try:
        # Robust parsing to handle potential trailing commas or spaces
        x_values = [float(val.strip()) for val in x_str.split(',') if val.strip()]
        y_values = [float(val.strip()) for val in y_str.split(',') if val.strip()]
        
        if not (x_values and y_values):
            st.error("Please enter data for both the X and Y axes."); return None, None, None, None
        if len(x_values) != len(y_values):
            st.error("The number of X values must match the number of Y values."); return None, None, None, None
    except ValueError:
        st.error("Invalid input. Please ensure all values are numbers."); return None, None, None, None

    x_np, y_np = np.array(x_values), np.array(y_values)
    
    # Extract unit symbols from full label strings (e.g., 'Time (s)' -> 's')
    x_unit = x_unit_label.split(' ')[-1].strip('()') if x_unit_label != "None" else ""
    y_unit = y_unit_label.split(' ')[-1].strip('()') if y_unit_label != "None" else ""
    
    # --- Matplotlib Plot Generation ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_np, y_np, marker='o', linestyle='', color='#1f77b4', markersize=7, label='Data Points')

    # Apply labels and title
    ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label, fontsize=12)
    ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label, fontsize=12)
    ax.set_title(f"Plot of {y_label} vs {x_label} (Basic Mode)", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Gradient calculation and plotting if requested
    grad_info = ""
    if show_gradient:
        grad, c, x1, y1, x2, y2, info = calculate_manual_gradient(x_np, y_np, x_unit, y_unit)
        if grad is not None:
            # Plot the fitted line across all x values
            ax.plot(x_np, c + grad * x_np, 'g--', linewidth=2, alpha=0.8, label=f'Manual Grad. Line (m: {grad:.4f})')
            # Highlight the two points used for the gradient
            ax.plot([x1, x2], [y1, y2], 's', color='r', markersize=10, markeredgecolor='black', label=f'Points Used: $P_1$ & $P_2$')
            grad_info = info
        else:
            grad_info = info # Pass error message
            
    ax.legend(loc='best')
    
    # Return figure and the calculated data/info
    return fig, grad_info, x_np, y_np

def run_basic_mode():
    """Main function to run the Basic Mode UI and logic."""
    st.markdown("Using **Basic Mode**.")
    st.info("This mode plots data and calculates a **Manual Gradient** using the two outermost points.")

    # Get inputs from the sidebar
    x_input, y_input, x_label, y_label, x_unit, y_unit, plot_triggered = data_input_sidebar_basic()
    
    loading_placeholder = st.empty()
    
    # Condition to trigger plotting logic:
    # 1. User clicks 'Plot Data' (plot_triggered is True)
    # 2. Plot state is empty, but we previously plotted (e.g., re-running the app, need to restore plot)
    # 3. User clicks 'Calculate Gradient' (show_gradient is True)
    trigger_plot_logic = plot_triggered or (st.session_state['fig'] is None and st.session_state['plotted']) or st.session_state['show_gradient']
    
    if trigger_plot_logic:
        # Show loading indicator only if a new action was triggered
        if st.session_state['show_gradient'] or plot_triggered:
            with loading_placeholder.container(): st.markdown("**Generating plot and analysis...**"); time.sleep(0.1)

        # Plot the graph and get results
        fig, grad_info, x_np, y_np = plot_basic_graph(x_input, y_input, x_label, y_label, x_unit, y_unit, st.session_state['show_gradient'])
        
        # Update session state with results
        st.session_state['fig'] = fig
        st.session_state['plotted'] = fig is not None
        st.session_state['show_gradient'] = False # Reset flag after plotting
        
        if fig is not None:
            st.session_state['gradient_markdown'] = grad_info
            # Store data for potential external use/debugging
            st.session_state['last_x_np'] = x_np
            st.session_state['last_y_np'] = y_np
            
        loading_placeholder.empty()

    # --- Display UI ---
    if st.session_state['fig']:
        st.pyplot(st.session_state['fig'])
        st.markdown("---")
        
        # Display gradient analysis if it was successfully calculated
        if st.session_state['gradient_markdown']:
            st.markdown(st.session_state['gradient_markdown'])
            st.markdown("---")
        
        col_calc, col_down, _ = st.columns([2, 2, 4])
        with col_calc:
            # Button to trigger the gradient calculation (setting the state flag)
            st.button("Calculate Gradient", on_click=set_gradient_state_local, type="primary")
        
        with col_down:
            # Download button
            buf = io.BytesIO()
            # Ensure the figure exists before trying to save it
            if st.session_state['fig']:
                 st.session_state['fig'].savefig(buf, format="png", bbox_inches='tight')
                 st.download_button("Download Plot", buf.getvalue(), "graPhyscs_plot.png", "image/png")
            
    elif not st.session_state['plotted']:
        st.info("Enter your data in the sidebar and click 'Plot Data' to begin.")
