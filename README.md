# PACCEVreserve

This project models and optimizes reserve-bid decisions for an EV fleet under stochastic arrivals and privacy-aware arrival-rate estimation. The workflow combines EV data preprocessing, synthetic record generation, private-rate inference, optimization, and validation.

## Project structure

- `src/data_reader.py`
  Loads and cleans the raw EV charging sessions and constructs the EV model parameters saved as JSON in `data/`.

- `src/lam_price_setting.py`
  Core utilities for model parameters, state space construction, arrival-rate estimation, and LDP randomized response.

- `src/EV_record_generator.py`
  Generates synthetic EV arrival records. It can create non-private records or LDP-private records with a chosen privacy budget.

- `src/DP_lambda_estimate.py`
  Estimates private and non-private arrival rates from EV records and saves them as a pickle file in `data/`.

- `src/SOC_deviation_model.py`
  Solves the reserve optimization model and produces policy outputs under different privacy settings.

- `src/SOC_bid_validation.py`
  Replays out-of-sample records to validate the policy and writes validation summaries into `results/`.

- `plot/SOC_bid_validation_plot.py`
  Reads validation summaries and produces comparison plots.

- `data/`
  Input datasets, EV model files, and generated synthetic EV record CSVs.

- `results/`
  Optimizer outputs and validation summaries.

- `job_opt_*.sh`
  Example LSF scripts for running the project on a cluster.

## Input data

The project expects market data files in `data/`:

- `data/fcrd_price.csv`
- `data/spot_price.csv`

The EV model is built from charging-session data and stored in JSON format such as:

- `data/ev_model_parameters_1.00_12.json`

This file contains the hourly arrival-rate profile `lambda_t` and the conditional EV state distribution used in the stochastic model.

## Environment setup

From the repository root:

```bash
python3 -m venv PACCEVreserve
source PACCEVreserve/bin/activate
pip install -r requirements.txt
```
## Running instructions

The project workflow is:

### 1) Build the EV model parameters

```bash
python src/data_reader.py
```

This reads the raw EV charging data and generates the parameter file used by the rest of the pipeline.

### 2) Generate synthetic EV records

Non-private baseline:

```bash
python src/EV_record_generator.py \
  --scale 100 \
  --output data/ev_records.csv \
  --insample-days 60 \
  --oos-days 60 \
  --seed 1042
```

LDP private records:

```bash
python src/EV_record_generator.py \
  --scale 100 \
  --DP LDP \
  --eps 10 \
  --output data/ev_records.csv \
  --insample-days 60 \
  --oos-days 30 \
  --seed 1042
```

The script writes CSV files such as:

- `data/ev_records_100_insample.csv`
- `data/ev_records_100_oos.csv`
- `data/ev_records_100_DP_eps10.0_insample.csv`
- `data/ev_records_100_DP_eps10.0_oos.csv`

### 3) Estimate lambda from noisy records

When running DP/LDP scenarios, estimate the arrival rates before the optimization step:

```bash
python src/DP_lambda_estimate.py --eps 10 --scale 100
```

This step is automatically triggered by the reserve optimization model if it has not already been done.

### 4) Run the reserve optimization model

Main optimization entry point:

```bash
python -u src/SOC_deviation_model.py --scale 100
```

LDP optimization:

```bash
python -u src/SOC_deviation_model.py --scale 100 --eps 10 --DP LDP
```

CDP / privacy variation runs follow the same structure, for example:

```bash
python -u src/SOC_deviation_model.py --scale 100 --eps 10 --DP CDP
```

The NoPP case is handled by commenting the required line in `src/SOC_deviation_model.py`, as noted in the script comments.

### 5) Validate the policy

After optimization, validate the output on out-of-sample records:

```bash
python -u src/SOC_bid_validation.py \
  --records data/ev_records_100_oos.csv \
  --input results/SOC_B_policy_results_100_0.9.pkl \
  --output results/SOC_B_bidvalidation_summary_records_100_0.9.csv
```

For a private run, use the matching policy file and output name. Example:

```bash
python -u src/SOC_bid_validation.py \
  --records data/ev_records_100_oos.csv \
  --input results/SOC_B_policy_results_DP_eps10.0_100_LDP_0.9.pkl \
  --output results/SOC_B_bidvalidation_summary_records_eps10._100_LDP_0.9.csv
```

This generates summary CSV files in `results/`.

### 6) Plot the validation results

```bash
python plot/SOC_bid_validation_plot.py
```

The plotting script reads the validation summaries and produces figures in the repository plot output directory.



## Typical end-to-end workflow

```bash
source PACCEVreserve/bin/activate
python3 src/data_reader.py
python src/EV_record_generator.py --scale 100 --output data/ev_records.csv --insample-days 60 --oos-days 60 --seed 1042
python src/DP_lambda_estimate.py --eps 10 --scale 100
python -u src/SOC_deviation_model.py --scale 100 --eps 10 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps10.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps10._100_LDP_0.9.csv
python plot/SOC_bid_validation_plot.py
```

## Notes

- The project uses Gurobi and expects a valid Gurobi license to be available.
- Run commands from the repository root.
- Output filenames vary with privacy setting, scale, and epsilon; check `results/` for the exact files produced by the latest run.
- Default settings assume a 24-hour horizon with 1-hour time steps unless you pass alternative CLI arguments.
- Default settings asusme beta = 0.1 (P90)
