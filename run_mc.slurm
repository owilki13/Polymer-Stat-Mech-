#!/bin/bash
#SBATCH --job-name=polymer
#SBATCH --output=logs/polymer_%A_%a.out
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-23

# ----------------------------
# Adjust this for your cluster
# ----------------------------

module purge

export PYTHONUNBUFFERED=1

python3 -u mc_polymer.py \
  --Ns 50 100 150 200 \
  --epsilons 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 \
  --steps 600000 \
  --burn 100000 \
  --sample-every 2000 \
  --max-disp 0.25 \
  --pivot-prob 0.03 \
  --temperature 1.0 \
  --init coil \
  --outdir mc_results