#!/bin/sh 
#BSUB -q hpc   # a queue to submit job to: hpc / gpua100 / gpuv100...
#BSUB -J job_inf-oue
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
### wall-time 24 hr
#BSUB -W 24:00 

### Doesnt overwrite, just makes new. 
#BSUB -o LSFlogs/%J_Output.out 
#BSUB -e LSFlogs/%J_Error.err

source PACCEVreserve/bin/activate	#!!! CREATE YOUR OWN "activate_venv.sh" with the correct path to your venv, and just keep it locally only.
module load python3/3.11.9   
module load pandas/2.2.2-python-3.11.9
module load scipy/1.13.0-python-3.11.9
module load  matplotlib/3.8.4-numpy-1.26.4-python-3.11.9  




#!/bin/bash
# set -e

SEEDS=(1043 1044 1045 1046 1222 1224 1226 1228 1230 1235 1240 1500 1550 1600 1650 3157 3333 3444 3465 3581 3651 3747 3846 3855 3912 4012 4038 4120 4238 4327 4820 4873 4876 5000 5100 5150 6122 6450 6710 6825 6840 7834 9650 36511 36522 47336 47895 51234 63205 71785 627282)
EPS_LIST=(8.0 4.0 2.0 1.0 0.5)

for SEED in "${SEEDS[@]}"; do
    OUTDIR="data/seed${SEED}"
    mkdir -p "$OUTDIR"

    for EPS in "${EPS_LIST[@]}"; do
        echo "=== seed ${SEED}, eps ${EPS} ==="

        # python -u src/EV_record_generator.py \
        #     --scale 100 --DP LDP --eps "$EPS" \
        #     --output "${OUTDIR}/ev_records.csv" \
        #     --insample-days 60 --oos-days 60 \
        #     --seed "$SEED"

        python src/DP_lambda_estimate.py \
            --eps "$EPS" --scale 100 --output "${OUTDIR}/lambda_estimate_eps${EPS}.pkl" \
            --seed "$SEED"
    done
done

#!/bin/bash
set -e

# SEEDS=(51234 627282 71785) #(2718 3141 4096 5381 6029 7919 8191 9973 10007 12345 20240 31337 42424 65537)

# for SEED in "${SEEDS[@]}"; do
#     OUTDIR="data/seed${SEED}"
#     mkdir -p "$OUTDIR"
#         python src/EV_record_generator.py --scale 100 --output data/seed${SEED}/ev_records.csv --insample-days 60 --oos-days 90 --seed ${SEED}
#     done
# done