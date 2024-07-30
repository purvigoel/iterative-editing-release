
for n in '0186' '0187' '0188' '0189' '0190' '0191' '0192' '0193' '0194' '0195' '0196' '0197' '0198' '0199' '0200' '0201' '0202' '0203' '0204' '0205' '0206' '0207' '0208' '0209' '0210' '0211' '0212' '0213' '0214' '0215' '0216' '0217' '0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' '0009' '0010' ; do 


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