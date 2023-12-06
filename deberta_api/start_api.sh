#!/bin/bash
#SBATCH --job-name=colbertapi
#SBATCH --output=colbertapi.out
#SBATCH --error=colbertapi.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00

python deberta_api.py

