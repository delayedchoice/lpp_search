#!/bin/bash
#
#SBATCH -J test_search_w_plots
#SBATCH -o ./SLURM_output/test_search_w_plots%j.txt

#SBATCH --account=2016394
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=45:00:00
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=general
#SBATCH --mail-user malharris19@unm.edu 
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --array=0-116 # for 116 files

# Change to the directory containing the Python script
cd ../src
module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python Executing_code_part_5_final_singles_search_and_figures.py $SLURM_ARRAY_TASK_ID

