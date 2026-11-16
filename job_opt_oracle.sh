#!/bin/sh
#BSUB -q hpc   # a queue to submit job to: hpc / gpua100 / gpuv100...
#BSUB -J oracle
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=200GB]"
#BSUB -M 200GB  ##400GB
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


## CDP 


SEEDS_test=(2718 3141 4096 5381 6029 7919 8191 9973 10007 12345 20240 31337 42424 65537)



i=0
for S in "${SEEDS_test[@]}"; do
    echo "=== test ${i} ==="
    #python -u src/SOC_deviation_model.py --scale 100 --eps ${EPS} --DP CDPnB --seed ${SEED}
    python -u src/SOC_bid_validation.py --records data/seed${S}/ev_records_100_oos.csv --input results/oracle/SOC_oracle_results_100_0.9.pkl --output results/test${i}/SOC_oracle_bidvalidation_summary_records_0.9.csv.csv
    i=$((i + 1))
done

#python -u src/SOC_deviation_model_oracle.py --scale 100 
#python -u src/SOC_bid_validation.py --input results/SOC_oracle_results_100_0.9.pkl --output results/SOC_oracle_bidvalidation_summary_records_0.9.csv --sample 1000 #--records data/ev_records_100_oos.csv


