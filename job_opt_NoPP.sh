#!/bin/sh
#BSUB -q hpc   # a queue to submit job to: hpc / gpua100 / gpuv100...
#BSUB -J CDP_NoPP
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=200GB]"
#BSUB -M 200GB
### wall-time 24 hr
#BSUB -W 30:00


### Doesnt overwrite, just makes new.
#BSUB -o LSFlogs/%J_Output.out
#BSUB -e LSFlogs/%J_Error.err

module load python3/3.11.9   
module load gurobipy/gurobi-12.0.1-python-3.11.9 
  
source EVCSAggreg/bin/activate

## REMEMBER TO UPDATE THE CODE BEFORE RUNNING THE FOLLOWING COMMANDS

## NO Privacy propagation

python -u src/SOC_B_deviation_model.py --scale 100 --eps 10 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps10.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps10._100_CDP_NoPP.csv

python3 -u src/SOC_B_deviation_model.py --scale 100 --eps 8 --DP CDP 
python3 -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps8.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps8.0_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 5 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps5.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps5.0_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 4 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps4.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps4.0_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 3 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps3.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps3.0_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 2 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps2.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps2.0_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 1 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps1.0_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps1.0_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 0.5 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps0.5_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps0.5_100_CDP_NoPP.csv

python -u src/SOC_B_deviation_model.py --scale 100 --eps 0.1 --DP CDP
python -u src/SOC_bidVal.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_DP_eps0.1_100_CDP_NoPP.pkl --output results/SOC_B_bidvalidation_summary_records_eps0.1_100_CDP_NoPP.csv
