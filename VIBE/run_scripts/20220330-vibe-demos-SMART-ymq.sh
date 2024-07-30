



cmd="python3 demo2.py \
	--vid_file=/scratch/users/wangkua1/data/smart/ymq01.mp4 \
	--output_folder=/scratch/users/wangkua1/data/smart/ymq01-vibe
"


if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=vibe-movi
#SBATCH --output=slurm_logs/vibe-movi-%j-out.txt
#SBATCH --error=slurm_logs/vibe-movi-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=1:00:00

#necessary env
# source /home/users/wangkua1/setup_rl.sh
echo \"$cmd\"
eval \"$cmd\"
"

fi

