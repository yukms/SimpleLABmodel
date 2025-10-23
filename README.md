---
title: LAB Model Alluvial Stratigraphy Simulator
emoji: "🟦"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.26.0"
app_file: app.py
pinned: false
---

# LAB Model Alluvial Stratigraphy Simulator

## Overview
This project simulates sedimentary basin stratigraphy using a simple LAB model. It incorporates aggradation, lateral migration, and periodic avulsion to generate stratigraphic layers. The simulation is visualized using Streamlit and provides analytical results such as Net-to-Gross (N/G) ratio, Amalgamation ratio, and Total Sand Bodies.

## Features
- **Interactive Parameter Control**: Adjust basin width, total time steps, channel width, avulsion period, lateral migration rate, and more using Streamlit sliders.
- **Scenario Switching**: Change parameters mid-simulation to observe the impact on stratigraphy.
- **Visualization**: View stratigraphy as a color-coded plot.
- **Analysis**: Get detailed statistics on sedimentary layers.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Labmodel
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the application in your browser at `http://localhost:8501`.
3. Adjust parameters in the sidebar and click "Run Simulation" to view results.

## Parameters
- **Basin Width**: Width of the basin in cells.
- **Total Time Steps**: Total simulation time.
- **Channel Width**: Width of the channel in cells.
- **Avulsion Period**: Time steps between channel jumps.
- **Lateral Migration Rate**: Rate of channel movement per time step.
- **Scenario Switch Step**: Time step at which parameters change.

## Results
- **Net-to-Gross (N/G) Ratio**: Proportion of sand to total sediment.
- **Amalgamation Ratio**: Vertical connectivity of sand layers.
- **Total Sand Bodies**: Number of connected sand regions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## Acknowledgments
This project uses the following libraries:
- **Streamlit**: For interactive web applications.
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting.
- **SciPy**: For image analysis.
