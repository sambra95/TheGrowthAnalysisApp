## Abstract

TheGrowthAnalysisApp is a Streamlit-based application for analyzing microbial growth curves from plate reader time-series data. Users upload raw optical density measurements and a plate map, configure analysis parameters, and the app computes baseline-corrected growth statistics such as maximum OD, growth rate, and lag time across wells. The workflow includes interactive quality control: users can review fitted phases, adjust boundaries, exclude wells, or re-fit data to ensure results reflect experimental context. The app then supports visualization and export of curated results for downstream reporting.

## Introduction

High-throughput plate reader experiments generate dense time-series data that require consistent preprocessing, fitting, and curation before biological conclusions can be drawn. TheGrowthAnalysisApp streamlines this process by combining structured data ingestion (plate maps and reader outputs), parameterized analysis (time clipping, pathlength correction, smoothing, and windowed growth fitting), and standardized outputs for downstream analysis.

Crucially, the app is designed for human-in-the-loop analysis. Automated fits provide a first-pass set of growth descriptors, but users can inspect each well, adjust phase boundaries, re-calculate growth rates from selected points, or mark wells as no-growth or excluded. This iterative review ensures the final dataset reflects both computational rigor and experimental expertise, enabling confident comparisons across strains, conditions, and replicates.

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- Git (optional, for cloning the repository)

### Setup Instructions

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/yourusername/TheGrowthAnalysisApp.git
   cd TheGrowthAnalysisApp
   ```

   Alternatively, download and extract the ZIP file from the repository.

2. **Create the conda environment**

   Use the provided `environment.yaml` file to create a conda environment with all required dependencies:

   ```bash
   conda env create -f environment.yaml
   ```

   This will create an environment named `growth_curves_app_env` with Python 3.11 and all necessary packages including:
   - Streamlit (web app framework)
   - NumPy, Pandas, SciPy (data processing and analysis)
   - Plotly (interactive visualizations)
   - openpyxl and xlrd (Excel file support)
   - Additional Streamlit components (plotly-events, sortables)

3. **Activate the environment**

   ```bash
   conda activate growth_curves_app_env
   ```

4. **Run the application**

   Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   The app will automatically open in your default web browser at `http://localhost:8501`.

- Ensure you're in the project directory when running `streamlit run app.py`
