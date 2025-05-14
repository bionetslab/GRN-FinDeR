#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --array=1-796%40  # Upper limit; will be adjusted dynamically
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
CONFIG_DIR="GRN-FinDeR/configs"
PROCESSED_DIR="GRN-FinDeR/processed_configs"
mkdir -p $PROCESSED_DIR  # Ensure the processed folder exists

TISSUES=($(ls $CONFIG_DIR/*.yaml | xargs -n 1 basename | sed 's/.yaml//'))

# Determine the number of tissues
NUM_TISSUES=${#TISSUES[@]}

# Ensure the array task ID is within the valid range
if [[ $SLURM_ARRAY_TASK_ID -gt $NUM_TISSUES ]]; then
    echo "No corresponding tissue for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Get the full filename (e.g., Heart_0)
TISSUE_FILE=${TISSUES[$SLURM_ARRAY_TASK_ID - 1]}

# Extract the tissue name (remove everything after the last underscore)
TISSUE=$(echo "$TISSUE_FILE" | sed 's/_[0-9]*$//')

echo "Processing tissue: $TISSUE (from file: $TISSUE_FILE.yaml)"

# Run the Python script for the specific YAML file
srun python GRN-FinDeR/classical_fdr_computation.py -f $CONFIG_DIR/${TISSUE_FILE}.yaml

# Ensure the output directory is based on the tissue name (without the index)
mkdir -p grn_finder_results/${TISSUE}

# Copy results
cp -r $TMPDIR/${TISSUE_FILE} grn_finder_results/${TISSUE}

# Move the processed config file to the new folder
cp $CONFIG_DIR/${TISSUE_FILE}.yaml $PROCESSED_DIR/

echo "Copied config file to $PROCESSED_DIR/${TISSUE_FILE}.yaml"

# Deactivate conda
conda deactivate