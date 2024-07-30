
for n in '0488' '0489' '0490' '0491' '0492' '0493' '0494' '0495' '0496' '0497' '0498' '0499' '0500' '0501' '0502' '0503' '0504' '0505' '0506' '0507' '0508' '0509' '0510' '0511' '0512' '0513' '0514' '0515' '0516' '0517' '0518' '0519' '0520' '0521' '0522' '0523' '0524' '0525' '0526' '0527' '0528' '0529' '0530' '2141' '2142' '2143' '2144' '2145' ; do 


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