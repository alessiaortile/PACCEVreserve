#!/bin/sh
#BSUB -q hpc   # a queue to submit job to: hpc / gpua100 / gpuv100...
#BSUB -J LDP_P90
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=200GB]"
#BSUB -M 200GB  
### wall-time 24 hr
#BSUB -W 30:00

### Doesnt overwrite, just makes new.
#BSUB -o LSFlogs/%J_Output.out
#BSUB -e LSFlogs/%J_Error.err



  
source PACCEVreserve/bin/activate
module load python3/3.11.9   
module load  matplotlib/3.8.4-numpy-1.26.4-python-3.11.9  
module load pandas/2.2.2-python-3.11.9
module load scipy/1.13.0-python-3.11.9
module load gurobipy/gurobi-12.0.1-python-3.11.9 

## B simulations ##
python -u src/SOC_deviation_model.py --scale 100
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_100_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_100_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 10 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps10.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps10._100_LDP_0.9.csv

python3 -u src/SOC_deviation_model.py --scale 100 --eps 8 --DP LDP
python3 -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps8.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps8_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 5 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps5.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps5.0_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 4 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps4.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps4.0_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 3 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps3.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps3.0_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 2 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps2.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps2.0_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 1 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps1.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps1.0_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 0.5 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps0.5_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps0.5_100_LDP_0.9.csv

python -u src/SOC_deviation_model.py --scale 100 --eps 0.1 --DP LDP
python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps0.1_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps0.1_100_LDP_0.9.csv
