# Benchmark

## How to generate configuration files:
1. Adjust template.yaml paths in src/template.yaml
2. Modify script to generate the correct SLURM submission!!!
3. Run ```python src/generate_configs_batch_scripts.py -f src/template.yaml -o configs/ -j jobscripts ```. This will create a bunch of slurm submission scripts.
4. ``` sbatch preprocess_all_data.sh ```
5. Update config scripts: ```python src/generate_configs_batch_scripts.py -o configs -u True ```
5. Run ``` sbatch ```
