


action=tennis_swing

cmd="python demo2.py \
	--vid_file=/home/groups/syyeung/wangkua1/data/mymocap/trimmed1/${action}.0.mp4 \
	--output_folder=/home/groups/syyeung/wangkua1/nemo/exps/mymocap_${action} \
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

