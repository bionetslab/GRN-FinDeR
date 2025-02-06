#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for Slurm version >22.05: cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo $SLURM_CPUS_PER_TASK
echo $SRUN_CPUS_PER_TASK
module load python
conda activate grn-finder 
cd $WORK 
srun python GRN-FinDeR/benchmark/src/generate_groundtruth.py -f GRN-FinDeR/benchmark/configs/Breast.yaml
cp $TEMPDIR/Breast /home/woody/iwbn/iwbn006h/grn_finder_results/Breast
conda deactivate
