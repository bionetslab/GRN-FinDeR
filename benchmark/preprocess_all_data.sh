#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --array=1-31%4  # Upper limit; will be adjusted dynamically
unset SLURM_EXPORT_ENV

# Set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SRUN CPUs per task: $SRUN_CPUS_PER_TASK"

# Load modules
module load python
conda activate grn-finder

# Change to working directory
cd $WORK

# Get the list of YAML files dynamically
CONFIG_DIR="GRN-FinDeR/benchmark/configs"
TISSUES=($(ls $CONFIG_DIR/*.yaml | xargs -n 1 basename | sed 's/.yaml//'))

# Determine the number of tissues
NUM_TISSUES=${#TISSUES[@]}

# Ensure the array task ID is within the valid range
if [[ $SLURM_ARRAY_TASK_ID -gt $NUM_TISSUES ]]; then
    echo "No corresponding tissue for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Get the tissue name for this task
TISSUE=${TISSUES[$SLURM_ARRAY_TASK_ID - 1]}

echo "Processing tissue: $TISSUE"

# Run the Python script for the specific tissue
srun python GRN-FinDeR/benchmark/src/preprocess_gtex.py -f $CONFIG_DIR/${TISSUE}.yaml

mkdir -p grn_finder_results/${TISSUE}
# Copy results

# Deactivate conda
conda deactivate

