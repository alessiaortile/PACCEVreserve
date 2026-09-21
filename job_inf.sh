#!/bin/sh 
#BSUB -q hpc   # a queue to submit job to: hpc / gpua100 / gpuv100...
#BSUB -J job_inf
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


python src/EV_recordGenerator.py --scale 100 --output data/ev_records.csv --insample-days 60 --oos-days 60 --seed 1042

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 10 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 10 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 8 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 8 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 5 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 5 --scale 100 

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 4 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 4 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 3 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 3 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 2 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 2 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 1 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 1 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 0.5 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 0.5 --scale 100

python src/EV_recordGenerator.py --scale 100 --DP LDP --eps 0.1 --output data/ev_records.csv --insample-days 60 --oos-days 30 --seed 1042
python src/Dp_laminfere.py --eps 0.1 --scale 100