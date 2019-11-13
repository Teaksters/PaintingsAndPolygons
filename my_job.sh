#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -t 120:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arneb.meijs@gmail.com

# Loading modules
module load 2019
module load Python/3.6.6-intel-2018b

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python main.py 
