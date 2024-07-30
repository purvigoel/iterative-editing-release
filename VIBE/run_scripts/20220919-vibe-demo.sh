



cmd="python demo2.py \
	--vid_file=/oak/stanford/groups/syyeung/hmr_datasets/h36m/S1/Videos/Walking.54138969.mp4 \
	--output_folder=/home/groups/syyeung/wangkua1/omomo/results/dev \
	--detector maskrcnn
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

echo \"$cmd\"
eval \"$cmd\"
"

fi

