#!/bin/bash
#SBATCH --job-name=build_mvd_x86_gcc_O1_O2    # Create a short name for your job, required
#SBATCH --nodes=1                             # Number of nodes, required
#SBATCH --ntasks=80                           # Number of CPU cores to use, required
#SBATCH --time=72:00:00                       # Time limit hrs:min:sec, required
#SBATCH --output build_mvd_x86_gcc_O1_O2.o%j  # Name of the stdout output file, required
#SBATCH --error build_mvd_x86_gcc_O1_O2.e%j   # Name of stderr error file 
#SBATCH --partition=compute2                  # Specify the name of the GPU partition, required

source ~/.bashrc
source $WORK/Repositories/vuln-decompiled-summarization/.venv/bin/activate  # Activate .venv from repository
export HF_HOME=$WORK/.cache/huggingface # You may not need to do these exports
export CONDA_PKGS_DIRS=$WORK/.conda/pkgs # Only if you're having disk quota exceeded errors

python $WORK/Repositories/vuln-decompiled-summarization/scripts/build_mvd.py $WORK/mvd_rebuilt.csv \
    "gcc -O1" "gcc -O2" \
    --ghidra ~/Ghidra/ghidra_11.0.1_PUBLIC/support/analyzeHeadless \
    --overwrite