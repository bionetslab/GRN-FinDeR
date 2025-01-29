#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH --export=NONE
module load python
conda activate alternet 
cd $WORK 
python GRN-FinDeR/benchmark/src/generate_groundtruth.py -f GRN-FinDeR/benchmark/configs/Spleen.yaml  
conda deactivate