#!/bin/bash
#SBATCH -A atm112
#SBATCH -J graph
#SBATCH -o logs/gnn-%j.o
#SBATCH -e logs/gnn-%j.e
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -q debug

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

# Activate your environment
source activate /lustre/orion/atm112/scratch/phu9/conda/torch

# Get address of head node
export MASTER_ADDR=$(hostname -i)

# Needed to bypass MIOpen, Disk I/O Errors
# export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache/node_${SLURM_NODEID}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Run script
##time srun python ./graph/main.py
time srun --gpus-per-task=1 --gpu-bind=closest python3 -W ignore -u ./graph/main.py