
for n in '0011' '0012' '0013' '0014' '0015' '0016' '0017' '0018' '0019' '0020' '0021' '0022' '0023' '0024' '0025' '0026' '0027' '0028' '0029' '0030' '0031' '0032' '0033' '0034' '0035' '0036' '0037' '0038' '0039' '0040' '0041' '0042' '0043' '0044' '0045' '0046' '0047' '0048' '0049' '0050' '0481' '0482' '0483' '0484' '0485' '0486' '0487' ; do 


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