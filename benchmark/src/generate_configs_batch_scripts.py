import yaml
import os.path as op
import os


def read_template(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_new_configs(path, config):
    tissues = ['Blood', 'Brain', 'Adipose Tissue', 'Muscle', 'Blood Vessel', 'Heart', 'Ovary', 'Uterus', 'Vagina',
               'Breast', 'Skin', 'Salivary Gland', 'Adrenal Gland', 'Thyroid', 'Lung', 'Spleen',
               'Pancreas', 'Esophagus', 'Stomach', 'Colon', 'Small Intestine', 'Prostate', 'Testis', 'Nerve',
               'Pituitary', 'Liver', 'Kidney', 'Cervix Uteri', 'Fallopian Tube', 'Bladder', 'Bone Marrow']
    for t in tissues:
        config['tissue'] = t
        name = t.replace(' ', '_')
        filename = op.join(path, f'{name}.yaml')
        with open(filename, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)


def save_jobscript(config, jobscript_path):

    '''
    Save the Slurm batch scripts.
    '''
    tissues = ['Blood', 'Brain', 'Adipose Tissue', 'Muscle', 'Blood Vessel', 'Heart', 'Ovary', 'Uterus', 'Vagina',
            'Breast', 'Skin', 'Salivary Gland', 'Adrenal Gland', 'Thyroid', 'Lung', 'Spleen',
            'Pancreas', 'Esophagus', 'Stomach', 'Colon', 'Small Intestine', 'Prostate', 'Testis', 'Nerve',
            'Pituitary', 'Liver', 'Kidney', 'Cervix Uteri', 'Fallopian Tube', 'Bladder', 'Bone Marrow']

    for t in tissues:
        config['tissue'] = t
        name = t.replace(' ', '_')
        jobscript_file = op.join(jobscript_path, f'{name}_script.sh')

        script_content = f"""#!/bin/bash -l
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
srun python GRN-FinDeR/benchmark/src/generate_groundtruth.py -f GRN-FinDeR/benchmark/configs/{t}.yaml
cp $TEMPDIR/{t} {config['results_dir']}/{t}
conda deactivate
"""

        with open(jobscript_file, 'w') as handle:
            handle.write(script_content)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process a file from the command line.")

    # Add the file argument
    parser.add_argument('-f', type=str, help='The file to process')
    parser.add_argument('-o', type=str, help='The output_directory')
    parser.add_argument('-j', type=str, help='Batch submissions scripts')

    # Parse the arguments
    args = parser.parse_args()
    config = read_template(args.f)
    os.makedirs(args.o, exist_ok=True)
    save_new_configs(args.o, config)
    os.makedirs(args.j, exist_ok=True)
    save_jobscript(config, args.j)




