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
set -e


SEEDS_train=(1042 1043 1044 1045 1046 1222 1224 1226 1228 1230 1235 1240 1500 1550 1600 1650 3157 3333 3444 3465 3581 3651 3747 3846 3855 3912 4000 4012
 4038 4120 4238 4327 4820 4873 4876 5000 5100 5150 6122 6450 6710 6825 6840 7834 9650 36511 36522 47336 47895 63205)
SEEDS_test=(2718 3141 4096 5381 6029 7919 8191 9973 10007 12345 20240 31337 42424 65537)
EPS_LIST=(8.0 4.0 2.0 1.0 0.5)


for SEED in "${SEEDS_train[@]}"; do
    OUTDIR="results/seed${SEED}"
    mkdir -p "$OUTDIR"

    for EPS in "${EPS_LIST[@]}"; do
        echo "=== seed ${SEED}, eps ${EPS} ==="
        i=0

        for S in "${SEEDS_test[@]}"; do
            echo "=== seed ${SEED}, eps ${EPS}, test ${i} ==="

            INPUT="results/seed${SEED}/SOC_B_policy_results_DP_eps${EPS}_100_LDP_0.9.pkl"
            if [ ! -f "$INPUT" ]; then
                echo "WARNING: missing $INPUT, skipping seed=${SEED}, eps=${EPS}"
                i=$((i + 1))
                continue
            fi

            python -u src/SOC_bid_validation.py \
                --records data/seed${S}/ev_records_100_oos.csv \
                --input results/seed${SEED}/SOC_B_policy_results_DP_eps${EPS}_100_LDP_0.9.pkl \
                --output results/seed${SEED}/test${i}/SOC_B_bidvalidation_summary_records_eps${EPS}_100_LDP_0.9.csv


            i=$((i + 1))
        done
    done
done

SEEDS=(1235 1240 1500 1550 1600 1650                               #1224 1226 1228 1230 
36511 36522 3855 3846 3912 3465 3444 3747 3651 3333
4120 4876 4238 4327 4038 4873 4012 4000 47895 47336
6840 63205 6450 6710 6122) #5000 5100 5150 6825 4820 9650 3581 3157 1222 7834
EPS_LIST=(8.0 4.0 2.0 1.0 0.5)

# for SEED in "${SEEDS[@]}"; do
#     OUTDIR="results/seed${SEED}"
#     mkdir -p "$OUTDIR"

#     for EPS in "${EPS_LIST[@]}"; do
#         echo "=== seed ${SEED}, eps ${EPS} ==="

#         python -u src/SOC_deviation_model.py --scale 100 --eps ${EPS} --DP LDP --seed ${SEED}
#         #python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/seed${SEED}/SOC_B_policy_results_DP_eps${EPS}_100_LDP_0.9.pkl --output results/seed${SEED}/SOC_B_bidvalidation_summary_records_eps${EPS}_100_LDP_0.9.csv
#     done
# done

#python -u src/SOC_deviation_model.py --scale 100
#python -u src/SOC_bid_validation.py --records data/ev_records_100_oos.csv --input results/SOC_B_policy_results_100_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_100_0.9.csv

#python -u src/SOC_deviation_model.py --scale 100 --eps 10 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps10.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps10._100_LDP_0.9.csv

#python3 -u src/SOC_deviation_model.py --scale 100 --eps 8 --DP LDP
#python3 -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps8.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps8.0_100_LDP_0.9.csv

#python -u src/SOC_deviation_model.py --scale 100 --eps 5 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps5.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps5.0_100_LDP_0.9.csv

# #python -u src/SOC_deviation_model.py --scale 100 --eps 4 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps4.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps4.0_100_LDP_0.9.csv

# #python -u src/SOC_deviation_model.py --scale 100 --eps 3 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps3.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps3.0_100_LDP_0.9.csv

# #python -u src/SOC_deviation_model.py --scale 100 --eps 2 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps2.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps2.0_100_LDP_0.9.csv

# #python -u src/SOC_deviation_model.py --scale 100 --eps 1 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps1.0_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps1.0_100_LDP_0.9.csv

# #python -u src/SOC_deviation_model.py --scale 100 --eps 0.5 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps0.5_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps0.5_100_LDP_0.9.csv

# #python -u src/SOC_deviation_model.py --scale 100 --eps 0.1 --DP LDP
#python -u src/SOC_bid_validation.py --input results/SOC_B_policy_results_DP_eps0.1_100_LDP_0.9.pkl --output results/SOC_B_bidvalidation_summary_records_eps0.1_100_LDP_0.9.csv
