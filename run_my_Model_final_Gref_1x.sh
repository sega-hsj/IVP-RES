#!/usr/bin/env bash
#SBATCH --job-name=my_Model-Gref-1x
#SBATCH --output=my_Model-Gref-1x.txt
#SBATCH --gres=gpu:4 -c 20 -x proj111,proj112,proj44

./train.sh