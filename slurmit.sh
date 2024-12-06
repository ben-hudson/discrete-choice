#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output /network/scratch/b/ben.hudson/slurm/%j.out
#SBATCH --error /network/scratch/b/ben.hudson/slurm/%j.err

set -e  # exit on error
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module --quiet purge
module --quiet load python/3.10
module --quiet load cuda/11.8

source .venv/bin/activate

if [ $# -lt 1 ]; then
    echo "Specify a script to run."
    exit 1
fi
script=${1}
args=${@:2}

# is this safe? no!
# does it matter? no!
python "${script}" --use_wandb --n_workers 8 ${args}
