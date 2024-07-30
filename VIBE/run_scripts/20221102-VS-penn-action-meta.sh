
for n in 0 1 2 3 4 5; do 


cmd="bash run_scripts/20221102-VS-penn-action-${n}.sh 0
"


if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=run-vibe.${n}
#SBATCH --output=slurm_logs/vibe-${n}-%j-out.txt
#SBATCH --error=slurm_logs/vibe-${n}-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu,syyeung
#SBATCH --time=1:00:00

echo \"$cmd\"
eval \"$cmd\"
"

fi

done