import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as nd_label


def run_lab_model(
    width=30,
    total_time=500,
    channel_width=10,
    avulsion_period=100,
    lateral_migration_rate=5,
    switch_step=None,
    new_channel_width=None,
    new_avulsion_period=None,
    new_lateral_migration_rate=None
):
    """
    Simple LAB Model: Uniform aggradation, lateral migration, and periodic avulsion.

    Args:
        width (int): Width of the basin cross-section (number of cells).
        total_time (int): Total simulation time (timesteps). Represents depth (Y-axis).
        channel_width (int): Width of the channel (sand) in cells.
        avulsion_period (int): Number of timesteps between avulsion (channel jump).
        lateral_migration_rate (float): Number of cells the channel moves per timestep (0~1).

    Returns:
        np.ndarray: 2D array of size (total_time, width).
                    0=none, 1=mud (floodplain), 2=sand (channel).
    """
    # Initialize variables
    stratigraphy = np.zeros((total_time, width), dtype=int)
    surface_elevation = np.zeros(width)
    channel_x = float(width) / 2  # Declare as float
    np.random.seed()  # System random
    migration_dir = np.random.choice([-1, 1])

    # Parameter change point (e.g., halfway through the simulation)
    if switch_step is None:
        switch_step = total_time // 2
    # Example of parameters to change (received as arguments from sliders)
    if new_avulsion_period is None:
        new_avulsion_period = avulsion_period
    if new_channel_width is None:
        new_channel_width = channel_width
    if new_lateral_migration_rate is None:
        new_lateral_migration_rate = lateral_migration_rate

    for t in range(total_time):
        # --- Change parameters mid-simulation ---
        if t == switch_step:
            avulsion_period = new_avulsion_period
            channel_width = new_channel_width
            lateral_migration_rate = new_lateral_migration_rate
        # --- 1. Uniform aggradation: Increase elevation of all cells by +1 ---
        surface_elevation += 1
        # --- 2. Channel lateral migration ---
        channel_x += migration_dir * lateral_migration_rate
        # Reverse direction and move once in the opposite direction if boundary is reached
        if channel_x <= channel_width / 2:
            channel_x = channel_width / 2
            migration_dir *= -1
            channel_x += migration_dir * lateral_migration_rate
        elif channel_x >= width - channel_width / 2 - 1:
            channel_x = width - channel_width / 2 - 1
            migration_dir *= -1
            channel_x += migration_dir * lateral_migration_rate
        # --- 3. Record stratigraphy ---
        new_layer_sed_type = np.full(width, 1)  # 1 = Mud
        start_x = max(0, int(np.floor(channel_x - channel_width / 2)))
        end_x = min(width, int(np.ceil(channel_x + channel_width / 2)))
        new_layer_sed_type[start_x:end_x] = 2  # 2 = Sand
        stratigraphy[t, :] = new_layer_sed_type
        # --- 4. Avulsion: Random channel jump every avulsion period ---
        if (t + 1) % avulsion_period == 0:
            channel_x = float(np.random.uniform(channel_width / 2, width - channel_width / 2))
            migration_dir = np.random.choice([-1, 1])
    return stratigraphy, switch_step

def analyze_stratigraphy(stratigraphy):
    """
    Analyze the generated stratigraphy and output simple statistics.
    """
    if stratigraphy.size == 0:
        print("No data to analyze.")
        return

    # N/G Ratio (Net-to-Gross): Proportion of sand (reservoir) in total sediment
    is_sand = (stratigraphy == 2)
    net_to_gross = np.sum(is_sand) / stratigraphy.size
    # Amalgamation Ratio (Vertical connectivity):
    sand_cells_below_first_row = is_sand[1:, :]
    if sand_cells_below_first_row.any():
        sand_above_sand = is_sand[1:, :] & is_sand[:-1, :]
        amalgamation_ratio = np.sum(sand_above_sand) / np.sum(sand_cells_below_first_row)
    else:
        amalgamation_ratio = 0
    labeled_array, num_features = nd_label(is_sand)
    print(f"--- Analysis Results ---")
    print(f"Net-to-Gross (N/G) Ratio: {net_to_gross:.2%}")
    print(f"Amalgamation Ratio (Vertical connectivity): {amalgamation_ratio:.2%}")
    print(f"Total Sand Bodies (Number of connected sand regions): {num_features}")
    print(f"------------------")

def plot_stratigraphy(stratigraphy, title, switch_step=None):
    colors = ['#8B4513', '#FFD700'] 
    cmap = ListedColormap(colors)
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(stratigraphy, aspect='auto', cmap=cmap, interpolation='none')
    plt.title(title)
    plt.xlabel("Basin Width (cells)")
    plt.ylabel("Time / Depth (timesteps)")
    plt.gca().invert_yaxis()
    if switch_step is not None:
        plt.axhline(switch_step, color='red', linewidth=2, linestyle='--', label='Scenario Switch')
    handles = [plt.Rectangle((0,0),1,1, color=colors[0]),
               plt.Rectangle((0,0),1,1, color=colors[1])]
    labels = ["Floodplain (Mud)", "Channel (Sand)"]
    plt.legend(handles, labels, loc='upper right')
    plt.show()

st.title("LAB Model Alluvial Stratigraphy Simulator")

# Sidebar: Parameter input
width = st.sidebar.slider("Basin Width (cells)", 10, 30, 20)
total_time = st.sidebar.slider("Total Time Steps", 100, 200, 100)
channel_width = st.sidebar.slider("Channel Width (cells)", 1, width//2, 5)
avulsion_period = st.sidebar.slider("Avulsion Period (timesteps)", 1, 10, 7)
lateral_migration_rate = st.sidebar.slider("Lateral Migration Rate (cells/timestep)", 0.1, 2.0, 1.0)

# Scenario switch parameters
switch_step = st.sidebar.slider("Scenario Switch Step", 1, total_time-1, total_time//2)
new_channel_width = st.sidebar.slider("New Channel Width", 1, width//2, 2)
new_avulsion_period = st.sidebar.slider("New Avulsion Period", 1, 10, 3)
new_lateral_migration_rate = st.sidebar.slider("New Lateral Migration Rate", 0.1, 2.0, 2.0)

if st.button("Run Simulation"):
    # Run simulation
    strat, switch_step_val = run_lab_model(
        width=width,
        total_time=total_time,
        channel_width=channel_width,
        avulsion_period=avulsion_period,
        lateral_migration_rate=lateral_migration_rate,
        switch_step=switch_step,
        new_channel_width=new_channel_width,
        new_avulsion_period=new_avulsion_period,
        new_lateral_migration_rate=new_lateral_migration_rate
    )
    # Apply values after parameter change point
    # switch_step, new_channel_width, etc., are already handled within run_lab_model

    # Output analysis results (before/after parameter change)
    st.subheader("Simulation Results")

    def analyze_segment(segment, label):
        is_sand = (segment == 2)
        net_to_gross = np.sum(is_sand) / segment.size
        sand_cells_below_first_row = is_sand[1:, :]
        if sand_cells_below_first_row.any():
            sand_above_sand = is_sand[1:, :] & is_sand[:-1, :]
            amalgamation_ratio = np.sum(sand_above_sand) / np.sum(sand_cells_below_first_row)
        else:
            amalgamation_ratio = 0
        labeled_array, num_features = nd_label(is_sand)
        return {
            "Label": label,
            "Net-to-Gross (N/G) Ratio": f"{net_to_gross:.2%}",
            "Amalgamation Ratio": f"{amalgamation_ratio:.2%}",
            "Total Sand Bodies": num_features
        }

    # Analyze results
    overall_results = analyze_segment(strat, "Overall Results")
    before_results = analyze_segment(strat[:switch_step_val, :], "Before Parameter Switch")
    after_results = analyze_segment(strat[switch_step_val:, :], "After Parameter Switch")

    # Display plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = ListedColormap(['#8B4513', '#FFD700'])
    ax.imshow(strat, aspect='auto', cmap=cmap, interpolation='none')
    ax.axhline(switch_step_val, color='red', linewidth=2, linestyle='--', label='Scenario Switch')
    ax.set_title("LAB Simulation")
    ax.set_xlabel("Basin Width (cells)")
    ax.set_ylabel("Time / Depth (timesteps)")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0,0),1,1, color='#8B4513'), plt.Rectangle((0,0),1,1, color='#FFD700')]
    labels = ["Floodplain (Mud)", "Channel (Sand)"]
    ax.legend(handles, labels, loc='upper right')
    st.pyplot(fig)

    # Display results in a table
    st.table([overall_results, before_results, after_results])



