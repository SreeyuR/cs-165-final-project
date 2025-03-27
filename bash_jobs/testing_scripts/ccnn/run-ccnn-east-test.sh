_#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=16G   # memory per CPU core
#SBATCH --time=0-04:00:00   # walltime format is DAYS-HOURS:MINUTES:SECONDS

#SBATCH -J "ccnn-east-train"   # job name
#SBATCH --mail-user=sratala@caltech.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

RUN_BASE="ccnn-east-train"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=$(printf "%s_%s" "$RUN_BASE" "$TIMESTAMP")

OUT_FILE="bash_jobs/runs/${RUN_NAME}.out"
ERR_FILE="bash_jobs/runs/${RUN_NAME}.err"

mkdir -p runs  # ensure the runs directory exists
exec >"$OUT_FILE" 2>"$ERR_FILE"

# modifies the PYTHONPATH env var to include path /groups/tensorlab/sratala/neuraloperator to find neuraloperator
# export PYTHONPATH="/groups/tensorlab/sratala/neuraloperator:$PYTHONPATH"
/central/groups/tensorlab/sratala/fusion/bin/python /groups/tensorlab/sratala/fno-disruption-pred/optuna_ccnn/testing/ccnn-east-test.py