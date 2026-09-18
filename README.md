# SWAP-OPTUNA

A Python-based framework for calibrating SWAP soil hydraulic parameters using Optuna, running SWAP simulations, and evaluating results against field observations.

## Table of Contents
- [SWAP-OPTUNA](#swap-optuna)
  - [Table of Contents](#table-of-contents)
  - [Project Identity](#project-identity)
  - [Baseline vs Calibration](#baseline-vs-calibration)
  - [Calibration Workflow](#calibration-workflow)
  - [Observation Matching Rule](#observation-matching-rule)
  - [Observation Dataset](#observation-dataset)
    - [Layer without field observations](#layer-without-field-observations)
  - [Diagnostic Outputs](#diagnostic-outputs)
  - [Performance Metrics](#performance-metrics)
  - [Plotting Capabilities](#plotting-capabilities)
    - [Global plots (layer-agnostic)](#global-plots-layer-agnostic)
    - [Layer-specific plots](#layer-specific-plots)
  - [Optuna Integration](#optuna-integration)
    - [How the optimization works](#how-the-optimization-works)
  - [Calibration Analysis](#calibration-analysis)
    - [Running the analysis](#running-the-analysis)
    - [Output structure](#output-structure)
    - [Generated outputs](#generated-outputs)
    - [Interpretation notes](#interpretation-notes)
  - [Current Calibration Result](#current-calibration-result)
    - [Best 50-trial TPE result (Trial 26)](#best-50-trial-tpe-result-trial-26)
    - [Legacy MATLAB baseline](#legacy-matlab-baseline)
  - [Project Structure](#project-structure)
  - [Generated and Ignored Files](#generated-and-ignored-files)
  - [Installation](#installation)
  - [Usage Instructions](#usage-instructions)
    - [Run calibration](#run-calibration)
    - [Run depth diagnostics](#run-depth-diagnostics)
    - [Run tests](#run-tests)
  - [Configuration Examples](#configuration-examples)
    - [Parameter bounds](#parameter-bounds)
    - [Requested diagnostic layers](#requested-diagnostic-layers)
    - [Requested VAP columns](#requested-vap-columns)
    - [Optuna trial configuration](#optuna-trial-configuration)
  - [Scientific Limitations](#scientific-limitations)
  - [Testing](#testing)
  - [Code Quality and Design Principles](#code-quality-and-design-principles)
## Project Identity

SWAP-OPTUNA is a Python-based framework for:
- Calibrating SWAP soil hydraulic parameters
- Running SWAP simulations
- Extracting depth-specific SWAP VAP outputs
- Comparing simulations with field observations
- Calculating statistical performance metrics
- Performing automated parameter optimization with Optuna
- Generating diagnostic visualizations

The project uses the SWAP model executable (Swap32.exe) together with Python orchestration and Optuna-based calibration. It is not a generic machine-learning project — it is specifically designed for soil hydraulic parameter calibration within the SWAP modelling framework.

## Baseline vs Calibration

This distinction is scientifically important.

The legacy MATLAB parameter set is a **baseline / reference parameterization**, not the output of the current Python/Optuna calibration. The current Optuna workflow is intended to calibrate the eight soil hydraulic parameters.

The legacy MATLAB parameterization and its associated workflow are described in the following peer-reviewed publication, which provides the methodological background for the reference parameterization:

**Noory, H., Khoshsimaie-Chenar, M., et al. (2025), "Developing a method for root-zone soil moisture monitoring at the field scale using remote sensing and simulation modeling."**  
DOI: https://doi.org/10.1016/j.agwat.2024.109263

The legacy MATLAB parameterization contains:

| Parameter | Description |
|-----------|-------------|
| ORES | Residual water content |
| OSAT | Saturated water content |
| ALFA | Air-entry related parameter |
| NPAR | Pore-size distribution index |
| KSATFIT | Saturated hydraulic conductivity (fitted) |
| LEXP | Pore-connectivity exponent |
| KSATEXM | Saturated hydraulic conductivity (extended) |
| BDENS | Bulk density |

Derived parameters:
- **ALFAW** = ALFA (wetting-curve alpha)
- **H_ENPR** = 0.0 (air-entry pressure head)

The current Python implementation defines the optimization bounds as:

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| ORES | 0.06 | 0.09 |
| OSAT | 0.25 | 0.50 |
| ALFA | 0.006 | 0.0082 |
| NPAR | 1.10 | 1.61 |
| KSATFIT | 10.01 | 20.00 |
| LEXP | 0.40 | 1.99 |
| KSATEXM | 20.01 | 40.01 |
| BDENS | 1350.01 | 1600.01 |

These values are the parameter search bounds implemented in `src/parameters.py`. Physical units are not explicitly documented in the source or model for these parameters.

**Important:** The current SWAP-OPTUNA Python framework is a separate implementation built on top of the SWAP model executable. It does not replicate the MATLAB codebase — the MATLAB parameterization serves as a baseline reference, and the Python workflow provides an Optuna-based calibration framework with isolated trial workspaces, exact-date observation matching, and VAP depth extraction capabilities that were not part of the original MATLAB implementation.

## Calibration Workflow

The actual pipeline implemented in the code is:

```
SoilParameters
      ↓
parameter validation
      ↓
SWAP template rendering
      ↓
isolated SWAP trial workspace
      ↓
Swap32.exe execution
      ↓
result.vap
      ↓
VAPReader
      ↓
depth-specific simulation extraction
      ↓
exact-date observation matching
      ↓
statistical metrics
      ↓
Optuna objective
      ↓
parameter optimization
```

Each Optuna trial is executed in an **isolated workspace**. The `SwapModel.create_trial_workspace()` method copies only the static reference files (templates, weather, crop, executable) into a new trial directory, excluding generated files. Generated SWAP files and results are kept separate from the reference model workspace.

The current objective minimizes **RMSE** (Root Mean Square Error). Additional metrics — MAE, MBE, R², NSE, and the number of matched observations — are stored as trial attributes via `trial.set_user_attr()`.

## Observation Matching Rule

Field observations are matched to SWAP simulation outputs using **EXACT DATE matching**:

- **No interpolation** between observation dates
- **No nearest-date matching**
- **No date shifting**
- **No temporal smoothing** for matching

Metrics are calculated **only** for dates present in both the observation and simulation datasets. The matcher uses an inner join on the `date` column with `validate="one_to_one"`, and the match is rejected if duplicate dates exist in either dataset.

**Why this matters:** the model is evaluated against actual field observations on their recorded dates. If a field observation was taken on 21-Apr-2019, the simulation is compared against the SWAP output for that exact date — not a nearby date or an interpolated value. Missing observation dates are not reconstructed.

## Observation Dataset

The tracked observation file is:

```
model/observation.csv
```

Its current structure is:

```csv
date,value
```


The current observations represent volumetric soil water content for the configured observation layer.

The current calibration/evaluation observation layer is:

```
0–5 cm
```


Field observations do not currently exist for all soil depths. The observation data consists of 10 dates recorded during the 2019 growing season:

| Date | Value (cm³/cm³) |
|------|-----------------|
| 10-Apr-2019 | 0.3176 |
| 21-Apr-2019 | 0.2992 |
| 26-Apr-2019 | 0.3140 |
| 03-May-2019 | 0.2950 |
| 12-May-2019 | 0.2620 |
| 14-May-2019 | 0.2910 |
| 19-May-2019 | 0.2770 |
| 27-May-2019 | 0.3266 |
| 07-Jun-2019 | 0.2504 |
| 20-Jun-2019 | 0.2440 |

The observation structure is `date,value` where `date` is a datetime and `value` is volumetric water content. No interpolation or reconstruction of missing dates is performed.

## SWAP VAP Depth Extraction

The functionality implemented by `src/vap.py` allows the reader to parse SWAP VAP output and extract depth-specific data. The reader can:

- Parse SWAP VAP output and ignore header/boundary records appropriately
- Identify actual soil compartments from the VAP file
- Define arbitrary requested soil layers using `RequestedLayer`
- Calculate overlap between requested layers and SWAP compartments
- Aggregate variables across requested layers with specified methods

The supported aggregation methods implemented in the code are:

| Method | Description |
|--------|-------------|
| `weighted_mean` | Thickness-weighted mean across compartments |
| `mean` | Arithmetic mean across selected compartments |
| `sum` | Sum across selected compartments |
| `top` | Value from the uppermost selected compartment |
| `bottom` | Value from the lowermost selected compartment |

Weighted means are based on compartment overlap thickness. The code supports only what is implemented — not all VAP variables have scientifically valid aggregation methods for every possible variable.

The VAP file format includes records with `date`, `depth`, and various soil variables (wcontent, phead, hconduc, temp, etc.), each associated with a soil compartment defined by `top` and `bottom` coordinates (in cm, positive downward, typically negative values in the VAP file).

## Multi-Layer Diagnostics

The script `run_depth_diagnostics.py` produces diagnostics for multiple configured soil layers. The current configured layers are:

| Layer | Depth Interval |
|-------|---------------|
| 0–5 cm | upper root zone |
| 5–10 cm | |
| 10–20 cm | |
| 20–30 cm | |
| 5–60 cm | deeper layer |

**These are CURRENT CONFIGURED DIAGNOSTIC LAYERS**, not universal defaults for SWAP or a scientific recommendation. Users can modify the `LAYERS` configuration in `run_depth_diagnostics.py` to request other depth intervals supported by the VAP extraction logic.

Current diagnostic variables are:

| Variable | Description |
|----------|-------------|
| wcontent | Volumetric water content |
| phead | Pressure head |
| hconduc | Hydraulic conductivity |
| temp | Temperature |

These are the currently configured diagnostic columns; the `COLUMNS` configuration controls which variables are extracted and exported for each layer.

## Observation Availability by Layer

The diagnostic workflow supports two situations:

### Layer with field observations

For example:

```
0–5 cm
```

The workflow produces:

- Simulation
- Field observations
- Exact-date matches
- Metrics (RMSE, MAE, MBE, R², NSE)
- Time-series plot
- Scatter plot (observed vs. simulated)

### Layer without field observations

The workflow still produces:

- Simulation
- CSV output
- Time-series plot

but does **NOT** calculate observational metrics or produce a scatter plot.

This distinction must be explicitly documented: a layer that has no corresponding field observation data will still generate simulation output and plots, but performance metrics and scatter plots are only generated when exact-date matches are found.

## Diagnostic Outputs

The output structure generated by `run_depth_diagnostics.py` is conceptually:

```
depth_diagnostics/
├── layer_0_5/
│   ├── simulation.csv
│   ├── time_series.png
│   └── scatter.png        # when observations/metrics exist
├── layer_5_10/
│   ├── simulation.csv
│   └── time_series.png
├── layer_10_20/
│   ├── simulation.csv
│   └── time_series.png
├── layer_20_30/
│   ├── simulation.csv
│   └── time_series.png
└── layer_5_60/
    ├── simulation.csv
    └── time_series.png
```

**`simulation.csv`** contains the requested simulation variables for that layer. The current configured CSV columns are:

```
date
wcontent
phead
hconduc
temp
```

These columns are controlled by the `COLUMNS` configuration — not all possible VAP columns are always exported. The simulation CSV includes a `layer` column indicating the depth interval, and the requested variables aggregated according to the specified methods.

## Performance Metrics

The metrics implemented in `src/metrics.py` are:

| Metric | Full Name | Definition |
|--------|-----------|------------|
| RMSE | Root Mean Square Error | `sqrt(mean((sim - obs)²))` |
| MAE | Mean Absolute Error | `mean(abs(sim - obs))` |
| MBE | Mean Bias Error | `mean(sim - obs)` (positive = overestimation, negative = underestimation) |
| R² | Squared Pearson correlation | Square of the Pearson correlation coefficient between observed and simulated |
| NSE | Nash–Sutcliffe Efficiency | `1 - SS_res / SS_tot` |
| n | Sample count | Number of matched observation/simulation pairs |

**Important:** The implementation calculates **R² as the square of the Pearson correlation coefficient** (see `MetricsCalculator.calculate()` in `src/metrics.py`), not as the coefficient of determination from a regression fit.

Metrics are calculated from **matched observation/simulation pairs** only. No qualitative labels such as "excellent", "good", or "poor" are assigned — the values are reported as-is for the user to interpret.

## Plotting Capabilities

Plotting capabilities are implemented in `src/plotting.py`. The module provides:

### Global plots (layer-agnostic)

- `plot_time_series()`: Continuous SWAP simulation and field observations on the same time axis
- `plot_scatter()`: Observed versus simulated values with 1:1 reference line

### Layer-specific plots

- `plot_layer_time_series()`: Time-series plot for one soil layer, showing simulation and optional observations
- `plot_layer_scatter()`: Observed-versus-simulated scatter plot for one soil layer with 1:1 line

Layer-specific plots can show:

- Continuous SWAP simulation
- Field observations when available
- Statistical metrics (RMSE, MAE, MBE, R², NSE, n) displayed as text on the plot
- Observed-versus-simulated scatter
- 1:1 reference line

**Plots do not prove model validity.** They are diagnostic tools for visual comparison. The metrics text included in plots reports calculated values without qualitative interpretation.

## Optuna Integration

The actual Optuna integration is configured as follows:

| Setting | Value |
|---------|-------|
| Study name | `swap_soil_calibration_tpe` |
| Sampler | `TPESampler` |
| Seed | `42` |
| Trials | `50` |
| Storage | SQLite database |
| Database | `optuna_study.db` |

### How the optimization works

- **Parameters are suggested within defined bounds** (see `src/parameters.py` `PARAMETER_BOUNDS`)
- **Each trial creates an isolated SWAP workspace** via `SwapModel.create_trial_workspace()` — the trial workspace is separate from the reference model and other trials
- **SWAP is executed** using the suggested parameter set
- **VAP output is evaluated** via `VAPReader` and `OutputEvaluator`
- **RMSE is returned as the optimization objective** (`direction="minimize"`)
- **Additional metrics are stored as trial attributes**: MAE, MBE, R², NSE, and n_matched are stored via `trial.set_user_attr()`

The current implementation is **single-objective optimization** (minimizing RMSE). The stored attributes allow users to inspect trade-offs after the optimization completes.

## Calibration Analysis

After an Optuna calibration run completes, the script `analyze_calibration.py` provides post-calibration diagnostics. It reads the stored Optuna study database and generates a structured set of CSV data files and figures that support evaluation of convergence, parameter sensitivity, identifiability, and equifinality.

This is a separate step from calibration — it does not run SWAP simulations or modify the Optuna study. It only reads existing results.

### Running the analysis

```powershell
python analyze_calibration.py
```

The script loads the Optuna study from `optuna_study.db`, processes all completed trials, and writes outputs to the `calibration_analysis/` directory.

### Output structure

```
calibration_analysis/
├── analysis_summary.txt
├── data/
│   ├── trial_results.csv
│   ├── top_trials.csv
│   ├── convergence_statistics.csv
│   ├── rmse_quantile_summary.csv
│   ├── parameter_importance_pedanova.csv
│   ├── parameter_importance_fanova.csv
│   ├── parameter_diagnostics.csv
│   ├── parameter_correlations.csv
│   ├── parameter_pair_correlations.csv
│   └── baseline_comparison.csv
└── figures/
    ├── optimization_history.png
    ├── parameter_importance.png
    ├── parallel_coordinate.png
    ├── parameter_correlations.png
    ├── slices/
    ├── contours/
    ├── convergence/
    ├── parameters/
    └── baseline/
```

### Generated outputs

| Output | Description |
|--------|-------------|
| **Convergence analysis** | Best-so-far RMSE curve, RMSE distribution histogram, and quantile summaries at selected fractions of top trials. Supports evaluation of whether the optimization has sufficiently explored the parameter space. |
| **Best-so-far RMSE tracking** | Trial-by-trial tracking of the best objective value, with relative improvement percentages. |
| **Parameter importance** | PedANOVA and fANOVA importance rankings for each calibrated parameter. These describe the relationship between sampled parameter values and the optimization objective within this study; they are not global sensitivity indices. |
| **Parameter correlations** | Pearson and Spearman correlations between each parameter and RMSE. These are screening diagnostics for identifying parameters most strongly associated with objective improvement. |
| **Parameter diagnostics** | Best-parameter normalized position within bounds, boundary proximity, and IQR of top-performing trials. Flags parameters whose best values lie within 5% of a bound. |
| **Parameter-pair correlations** | Pairwise Pearson correlations computed for all trials, top 10%, and top 20%. Supports identification of parameter interactions and equifinality patterns. |
| **Top trial summaries** | Distribution statistics (min, Q1, median, Q3, max, IQR, std) of parameter values among the top 10 trials. |
| **Baseline comparison** | Side-by-side comparison of Optuna best-trial metrics with the legacy MATLAB baseline (RMSE, MAE, MBE, R², NSE). The comparison is descriptive and does not establish statistical superiority of either method. |
| **Calibration diagnostics** | Normalized parameter positions within bounds, top-trial parameter distributions, and contour plots of parameter interactions. |

### Interpretation notes

All outputs are diagnostic — they describe the behavior of this particular Optuna study and do not constitute proof of model validity or parameter identifiability. Specific caveats:

- Parameter importance rankings from fANOVA and PedANOVA use different statistical frameworks and may produce different orderings.
- Boundary proximity indicates proximity to a predefined calibration bound; it does not by itself justify modifying the bound.
- Top-trial parameter ranges are empirical distributions among the best sampled trials, not formal confidence intervals.
- Correlation screening is not equivalent to formal global sensitivity analysis.

## Current Calibration Result

A comparison between the current 50-trial TPE run and the legacy MATLAB baseline:

### Best 50-trial TPE result (Trial 26)

| Metric | Value |
|--------|-------|
| RMSE | 0.027669 |
| MAE | 0.022460 |
| MBE | 0.008900 |
| R² | 0.467819 |
| NSE | -0.038607 |
| n | 10 |

Parameters:

| Parameter | Value |
|-----------|-------|
| ORES | 0.0866921862826 |
| OSAT | 0.3962009762 |
| ALFA | 0.00667620209588 |
| NPAR | 1.10495584823 |
| KSATFIT | 18.893325897 |
| LEXP | 1.01998176493 |
| KSATEXM | 33.3711363382 |
| BDENS | 1419.45716241 |

### Legacy MATLAB baseline

| Metric | Value |
|--------|-------|
| RMSE | 0.027357 |
| MAE | 0.021560 |
| MBE | -0.013840 |
| R² | 0.485691 |
| NSE | -0.015290 |
| n | 10 |

These values are presented as a **factual comparison** between the current 50-trial TPE run and the legacy MATLAB baseline. No claim is made that the Optuna calibration outperformed MATLAB or that the Python implementation is superior.

## Project Structure

```
SWAP-OPTUNA/
├── .gitignore
├── pyproject.toml
├── README.md
├── run_calibration.py
├── analyze_calibration.py
├── run_depth_diagnostics.py
│
├── src/
│   ├── calibration.py
│   ├── evaluation.py
│   ├── matching.py
│   ├── metrics.py
│   ├── observations.py
│   ├── objective.py
│   ├── parameters.py
│   ├── plotting.py
│   ├── swap.py
│   └── vap.py
│
├── model/
│   ├── CROP.crp
│   ├── heatparam.csv
│   ├── soilphysparam.csv
│   ├── swap.ok
│   ├── SWAP.swp
│   ├── Swap32.exe
│   ├── swap_swap.log
│   ├── SWAP_template.txt
│   ├── weather.018
│   ├── weather.019
│   ├── observation.csv
│   └── results/
│
└── tests/
    ├── test_calibration.py
    ├── test_evaluation.py
    ├── test_matching.py
    ├── test_metrics.py
    ├── test_observations.py
    ├── test_parameters.py
    ├── test_plotting.py
    ├── test_swap.py
    └── test_vap.py
```

## Generated Files

Generated files/directories:

| File/Directory | Description |
|----------------|-------------|
| `model/SWAP.swp` | Generated SWAP input file |
| `model/swap.ok` | SWAP completion flag |
| `model/swap_swap.log` | SWAP execution log |
| `model/heatparam.csv` | Generated heat parameters |
| `model/soilphysparam.csv` | Generated soil physics parameters |
| `model/results/*` | SWAP simulation results |
| `optuna_runs/` | Optuna trial workspaces |
| `optuna_study.db` | Optuna study database |
| `baseline_results/` | Baseline evaluation results |
| `best_results/` | Best trial results directory |
| `depth_diagnostics/` | Depth-layer diagnostic outputs |
| `calibration_analysis/` | Post-calibration diagnostic outputs from `analyze_calibration.py` |


## Installation

The project requires Python >=3.12. Dependencies are defined in `pyproject.toml`:

```
[project]
name = "swap-optuna"
version = "0.1.0"
description = "SWAP soil hydraulic parameter calibration"
requires-python = ">=3.12"
```

**SWAP requires a Windows executable** (`Swap32.exe`), so a Windows runtime is necessary to run the model.

To install the Python package and dependencies in development mode:

```powershell
pip install -e .
```

This installs the package from the source directory and makes the modules available from `src/`.

## Usage Instructions

Provide practical commands for the current workflow.

### Run calibration

```powershell
python run_calibration.py
```

This runs the Optuna calibration and evaluates the best parameter set. The study uses TPESampler with seed 42 and 50 trials, storing results in `optuna_study.db`.

### Analyze calibration results

```powershell
python analyze_calibration.py
```

This reads the existing Optuna study from `optuna_study.db` and generates post-calibration diagnostics in the `calibration_analysis/` directory. It does not run additional SWAP simulations. The analysis includes convergence tracking, parameter importance rankings, parameter–RMSE correlations, top-trial summaries, and a comparison against the legacy MATLAB baseline.

### Run depth diagnostics

```powershell
python run_depth_diagnostics.py
```

This reads `model/results/result.vap` and produces layer-specific diagnostics in the `depth_diagnostics/` directory. The configured layers are 0–5 cm, 5–10 cm, 10–20 cm, 20–30 cm, and 5–60 cm. The diagnostic variables are wcontent, phead, hconduc, and temp (as configured in `COLUMNS`).

If a layer has no field observations, the workflow still produces a simulation CSV and time-series plot, but does not calculate metrics or produce a scatter plot.

### Run tests

```powershell
pytest -v
```

This runs the test suite. Integration tests (those that execute the SWAP model) are marked with `@pytest.mark.integration` and can be run with `pytest -m integration`.

## Configuration Examples

Users can modify the following in the actual source files:

### Parameter bounds

Edit `src/parameters.py` — `PARAMETER_BOUNDS` dictionary:

```python
PARAMETER_BOUNDS = {
    "ores": (0.06, 0.09),
    "osat": (0.25, 0.50),
    "alfa": (0.006, 0.0082),
    "npar": (1.10, 1.61),
    "ksatfit": (10.01, 20.00),
    "lexp": (0.40, 1.99),
    "ksatexm": (20.01, 40.01),
    "bdens": (1350.01, 1600.01),
}
```

### Requested diagnostic layers

Edit `run_depth_diagnostics.py` — `LAYERS` list:

```python
LAYERS = [
    RequestedLayer(0, 5),
    RequestedLayer(5, 10),
    RequestedLayer(10, 20),
    RequestedLayer(20, 30),
    RequestedLayer(5, 60),
]
```

### Requested VAP columns

Edit `run_depth_diagnostics.py` — `COLUMNS` list:

```python
COLUMNS = [
    RequestedColumn("wcontent", "weighted_mean"),
    RequestedColumn("phead", "weighted_mean"),
    RequestedColumn("hconduc", "weighted_mean"),
    RequestedColumn("temp", "weighted_mean"),
]
```

### Optuna trial configuration

Edit `run_calibration.py` — `STUDY_NAME`, `N_TRIALS`, `DATABASE_PATH`:

```python
STUDY_NAME = "swap_soil_calibration_tpe"
N_TRIALS = 50
DATABASE_PATH = PROJECT_DIR / "optuna_study.db"
```

### Analysis configuration

Edit `analyze_calibration.py` — `STUDY_NAME`, `STORAGE`, `TOP_N`, `TOP_IMPORTANCE_PARAMETERS`:

```python
STUDY_NAME = "swap_soil_calibration_tpe"
STORAGE = "sqlite:///optuna_study.db"
TOP_N = 10
TOP_IMPORTANCE_PARAMETERS = 4
```

The fixed constants `PARAMETERS` and `PARAMETER_BOUNDS` must match the parameter names and bounds used by the calibration study.

## Scientific Limitations

1. **Current calibration evaluates the configured 0–5 cm observation layer.** Field observations are only available for this layer; deeper layers are simulated but cannot be calibrated against field data.

2. **Field observations are sparse in time.** The observation file contains 10 dates from April to June 2019. Metrics based on such a small sample should be interpreted with caution.

3. **Evaluation uses exact-date matching.** Only dates present in both the observation and simulation datasets contribute to the metrics. No interpolation or date adjustment is applied.

4. **A layer without field observations cannot receive observational performance metrics.** Simulation results are still extracted, but statistical evaluation is not possible without observed data for that layer.

5. **Multi-depth simulation differences do NOT by themselves prove soil heterogeneity.** Differences in extracted layer values are the result of the physical model and coordinate grid; they should not be over-interpreted without additional evidence.

6. **The current SWAP template applies the same hydraulic parameter set to the configured soil layers.** The repository does not implement independently calibrated depth-specific hydraulic parameters, and no such claim should be inferred from the code.

7. **Calibration quality depends on SWAP model structure, parameter bounds, observations, forcing data, initial conditions, and optimization configuration.** The current setup reflects one particular modelling configuration.

## Testing

The test suite is located in `tests/`. The tests cover:

- **Parameter validation** — bounds checking in `src/parameters.py`
- **SWAP input handling** — template rendering and input file generation in `src/swap.py`
- **VAP parsing and aggregation** — file reading, layer extraction, and aggregation methods in `src/vap.py`
- **Exact observation matching** — date-based matching rules in `src/matching.py`
- **Metric calculation** — RMSE, MAE, MBE, R², NSE in `src/metrics.py`
- **Plotting** — time-series and scatter plot generation in `src/plotting.py`
- **Multi-layer evaluation** — layer extraction and evaluation in `src/evaluation.py`
- **Depth diagnostics** — the diagnostic pipeline in `run_depth_diagnostics.py`

The integration tests (those that execute the SWAP model) are marked with `@pytest.mark.integration` and are excluded by default (`addopts = "-m 'not integration'"` in `pyproject.toml`). These can be run explicitly with:

```powershell
pytest -m integration
```

The full suite (except integration tests) can be run with:

```powershell
pytest -v
```

Coverage is not claimed as a percentage; the reported tests are a subset of what the repository implements.

## Code Quality and Design Principles

The codebase exhibits the following structural characteristics:

- **Modular source files** — functionality is separated into `src/` modules (`parameters.py`, `swap.py`, `vap.py`, `evaluation.py`, `metrics.py`, `matching.py`, `plotting.py`, etc.).
- **Dataclasses for structured parameters/results** — e.g., `SoilParameters`, `RegressionMetrics`, `CalibrationResult`, `LayerEvaluationResult`, `RequestedLayer`, `RequestedColumn`.
- **Separation of SWAP execution from evaluation** — `SwapModel` handles model running; `CalibrationRunner` and `OutputEvaluator` handle evaluation.
- **Exact-date matching** — a scientific rule enforced by `ObservationMatcher`.
- **Isolated Optuna trial workspaces** — each trial runs in its own directory.
- **Test-driven incremental development** — the test suite covers core logic.
- **Backward-compatible plotting functions** — `plot_time_series`/`plot_scatter` behave the same for existing code, while new layer-specific functions provide extended diagnostics.

These are observed characteristics of the current implementation, not formal guarantees.

**This repository represents an early research prototype and is under active development.**
