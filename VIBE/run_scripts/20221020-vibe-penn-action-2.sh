
for n in '0044' '0045' '0046' '0047' '0048' '0049' '0050' ; do 


cmd="python demo2.py \
	--input_image_folder=/oak/stanford/groups/syyeung/wangkua1/data/data/Penn_Action/frames/${n} \
	--output_folder=/oak/stanford/groups/syyeung/wangkua1/data/data/Penn_Action/vibe_results \
	--detector maskrcnn \
	--render 0
"


if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${action}.${n}
#SBATCH --output=slurm_logs/vibe-${action}.${n}-%j-out.txt
#SBATCH --error=slurm_logs/vibe-${action}.${n}-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=1:00:00

echo \"$cmd\"
eval \"$cmd\"
"

fi

done