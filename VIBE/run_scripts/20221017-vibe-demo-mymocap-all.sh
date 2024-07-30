for n in 0 1 2 3 4 5 6 7; do

partition=gpu,syyeung

cmd="python -m scripts.20221017_run_vibe \
    --n $n
"

if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=run-vibe-${partition}
#SBATCH --output=slurm_logs/run-vibe-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/run-vibe-${partition}-%j-err.txt
#SBATCH --mem=32gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p ${partition}
#SBATCH --time=48:00:00

#necessary env
# source /home/users/wangkua1/setup_rl.sh
echo \"$cmd\"
eval \"$cmd\"
"

fi
done