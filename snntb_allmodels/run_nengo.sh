#!/bin/bash
#
#SBATCH --job-name=power_measurement
#
#SBATCH --output=./out/power_measurement_%j.out
#SBATCH --error=./out/power_measurement_%j.err
#
#SBATCH --export=ALL
#
#SBATCH --nodes=2

### source ~/.bashrc

### source ./homes/smithmh6/env/scripts/activate

PARTITION=loihi python nengo_allmodels.py
