#!/bin/bash
#SBATCH --job-name=polymer
#SBATCH --output=logs/polymer_%A_%a.out
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-47%8



module purge
module load lammps

BASE=${SLURM_SUBMIT_DIR}

mkdir -p "${BASE}/logs"
mkdir -p "${BASE}/md_runs"

Ns=(50 100 150 200)
EPS=(0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20)

n_eps=${#EPS[@]}

i=${SLURM_ARRAY_TASK_ID}
n_index=$(( i / n_eps ))
e_index=$(( i % n_eps ))

if [ $n_index -ge ${#Ns[@]} ]; then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

N=${Ns[$n_index]}
EPSILON=${EPS[$e_index]}
SEED=$((1000 + N + e_index))

WORKDIR="${BASE}/md_runs/N${N}_eps${EPSILON}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}" || exit 1

cp "${BASE}/in.polymer_md" .
cp "${BASE}/make_lammps_chain.py" .

python3 -u make_lammps_chain.py \
    --N "${N}" \
    --mode coil \
    --seed "${SEED}" \
    --min-dist 0.90 \
    --out polymer.data

stdbuf -oL -eL srun lmp -in in.polymer_md \
    -var datafile polymer.data \
    -var eps "${EPSILON}" \
    -var temp 1.0 \
    -var dt 0.0005 \
    -var nsteps 2000000 \
    -var seed "${SEED}"