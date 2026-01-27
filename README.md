## Abstract

TheGrowthAnalysisApp is a Streamlit-based application for analyzing microbial growth curves from plate reader time-series data. Users upload raw optical density measurements and a plate map, configure analysis parameters, and the app computes baseline-corrected growth statistics such as maximum OD, maximum specific growth rate, and lag time across wells. The workflow includes interactive quality control: users can review fitted phases, adjust boundaries, exclude wells, or re-fit data to ensure results reflect experimental context. The app then supports visualization and export of curated results, including auto-detection and averaging of replicates, for downstream reporting.

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

2. **Create the conda environment**

   Use the provided `environment.yaml` file to create a conda environment with all required dependencies:

   ```bash
   conda env create -f environment.yaml
   ```

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

## How to use the app

### 1) Prepare your input files

- **Plate reader data**: a time-series table with time in the first column and OD measurements for each well in subsequent columns.
- **Plate map**: a table that maps wells to sample metadata (strain, condition, replicate, etc.).

Sample files are included in the repo:

- `example_data.xlsx`
- `example_plate_map.xls`

### 2) Upload and configure

1. Open the app in your browser.
2. Go to the **Upload & Analyse** page.
3. Upload the raw data file and the plate map.
4. Configure analysis settings such as:
   - time window (clip start/end)
   - pathlength correction and baseline subtraction
   - smoothing options
   - fitting window or growth phase selection

### 3) Review and curate fits

- Inspect growth curves and fitted phases.
- Adjust phase boundaries if needed.
- Mark wells as excluded or no-growth.
- Re-fit data based on selected points.

### 4) Export results

- Export the curated summary statistics (max OD, growth rate, lag time, etc.).
- Export curated plots for reporting.

## Tips

- If uploads fail, confirm your files are Excel-compatible and have consistent well names.
- Start with the example files to validate the workflow before using your own data.
