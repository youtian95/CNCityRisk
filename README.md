# Probabilistic Seismic Resilience Assessment of Major Chinese Cities

[中文版](readme_zh.md)

## Overview

This project assesses the seismic resilience of major Chinese cities and provides the associated data and analysis results.

## Analysis Workflow

### Standard City Analysis

1. Set `CityNames` in `Main.py` to the Chinese names of the cities to analyse, without the suffix “市”. For example:

   ```python
   CityNames = ['北京']
   ```

2. Run `Main.py`. Analysis results are generated in `CNCityRisk/results/`.

3. After running `Utilities.package_web_static_map_data()`, copy the five generated folders (`AnnualizedRisk`, `IMmap`, `mbtiles`, `RegionalLossStatistics`, and `Ruptures`) into `CNCityRiskWeb/static/maps/` to view the results in the web application.

### Special Case: Validation Against the 2008 Wenchuan Earthquake

The `reproduce_wenchuan2008_event()` function in `Main.py` reproduces the regional analysis used for validation against the 2008 Wenchuan earthquake.

## Project Structure

- `config.ini`: Configuration file containing API keys. Create this file manually with the following contents:

  ```ini
  [API]
  api_key_OpenTopography = <YOUR API KEY>
  ```

- `README.md`: Project overview, workflow, directory structure, and setup instructions in English.
- `readme_zh.md`: Original Chinese documentation.
- `Main.py`: Main analysis workflow.
- `requirements.txt`: Python dependencies.
- `CNCityRisk/`: Data processing and analysis scripts.
  - `Data/`: Raw and processed input data.
    - `CNBldFootprintHeight_Che2024`: Source data for building footprints and heights.
    - `POI_AMAP_2020`: Amap POI data used to predict structural types.
    - `GEM-CHN-v2021.1.0`: Seismic source model used to generate earthquake scenarios.
  - `results/`: Intermediate and final analysis results.
    - `BldData`: Processed building attributes, including height, area, and structural type, used to generate structural models.
    - `AnnualizedRisk`: Annualised loss calculations. **Do not delete.**
    - `EQsources`: Seismic source and earthquake event data. **Do not delete.**
    - `IDA_Results`: Incremental dynamic analysis (IDA) results, with one file per representative structure. **Do not delete.**
    - `IMsim`: Simulated ground-motion intensity fields and intensities interpolated at building locations. Can be deleted after all analyses for a city are complete.
    - `IndBldLoss`: Simulated seismic losses for individual buildings. **Do not delete.**
    - `Maps`: Map data generated for web visualisation. **Do not delete.**
    - `RegionalLossSim`: Regional loss simulation results. Can be deleted after all analyses for a city are complete.
    - `ReprBld`: Structural model parameters for representative structures used in IDA. **Do not delete.**
    - `ReprBld_PSPA`: Representative building information used in individual-building probabilistic seismic loss assessment. **Do not delete.**
    - `temp`: Temporary and intermediate files.
  - `PrepareBldData.py`: Prepares basic building attributes.
  - `plot_city.py`: Plots city regions.
  - `ClassifyBld.py`: Classifies buildings and selects representative buildings.
  - `IDA_ReprBld.py`: Performs IDA for representative structures.
  - Other analysis modules.
  - `Utilities.py`: Utility functions.

## Development Environment

1. **Create a virtual environment.** Open Anaconda Prompt or a terminal and run:

   ```bash
   conda create -n quake_resilience python=3.12 numpy==1.26.2
   ```

   NumPy is pinned to version 1.26.2 because other versions may conflict with `openquake.engine` in this environment.

2. **Activate the environment:**

   ```bash
   conda activate quake_resilience
   ```

3. **Install dependencies:**

   ```bash
   pip install openquake.engine BldStructPred MDOFModel BldCluster h5py folium cartopy dask geojsoncontour addressparser pypinyin py7zr rasterio cnmaps
   ```

4. **Install this project in editable mode** from the repository root:

   ```bash
   pip install -e .
   ```

5. **Run the project:**

   ```bash
   python Main.py
   ```
