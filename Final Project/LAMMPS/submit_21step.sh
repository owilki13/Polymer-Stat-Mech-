#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --partition=mit_normal

module load openmpi

mpirun -np $SLURM_NTASKS /orcd/pool/008/zpsmith_shared/software/lammps-29Aug2024/src/lmp_mpi -in 21step.in
