#!/bin/bash
#SBATCH --job-name=trapez
#SBATCH -n 4
#SBATCH --output=output_trapez.txt
hostname
gcc -fopenmp omp_trapezAreaSimple.C -o omp
./omp