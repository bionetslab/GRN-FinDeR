#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --array=1-99%60  # Adjust manually or with a helper script

unset SLURM_EXPORT_ENV

module purge 
source /home/woody/iwbn/iwbn106h/software/miniforge3/etc/profile.d/conda.sh
conda activate grn2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$WORK"

CONFIG_DIR="$WORK/configs_random_breast_kidney_testis"
PROCESSED_DIR="$WORK/configs_random_breast_kidney_testis_processed"
mkdir -p "$PROCESSED_DIR"

# Dynamically build list of config files (sorted for consistency)
mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' | sort)

NUM_CONFIGS=${#CONFIG_FILES[@]}

# Check SLURM array bounds
if (( SLURM_ARRAY_TASK_ID > NUM_CONFIGS )); then
    echo "No config file for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Map task ID to config file (0-based array vs 1-based SLURM index)
CONFIG_FILE="${CONFIG_FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "Processing config: $CONFIG_FILE"

# Extract tissue name from file (remove path and trailing _digits)
#CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
#TISSUE="${CONFIG_BASENAME%%_[0-9]*}"

#echo "Processing tissue: $TISSUE"

# Run your processing
srun python /home/hpc/iwbn/iwbn106h/Projects/GRN-FinDeR.git/approximate_fdr_computation.py -f "$CONFIG_FILE"

# Copy results
#mkdir -p "grn_finder_results/${TISSUE}"
#cp -r "$TMPDIR/${CONFIG_BASENAME}" "grn_finder_results/${TISSUE}"

# Move processed config
cp "$CONFIG_FILE" "$PROCESSED_DIR/"

echo "Moved config file to processed directory."

