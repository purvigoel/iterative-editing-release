

# for vid_id in  {9..128}; do
# for vid_id in  9; do
for vid_id in  {9..128}; do

fname=$(ls /scratch/users/wangkua1/data/S1/Videos/  | sed -n ${vid_id}p)
fname=$(printf %q "$fname") # Escapae whitespaces

cmd="python demo.py \
	--vid_file=/scratch/users/wangkua1/data/S1/Videos/${fname} \
	--output_folder=/scratch/users/wangkua1/results/vibe/s1/${fname}
"

if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=vibe-demos3
#SBATCH --output=slurm_logs/vibe-demos3-%j-out.txt
#SBATCH --error=slurm_logs/vibe-demos3-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p syyeung
#SBATCH --time=1:00:00

#necessary env
# source /home/users/wangkua1/setup_rl.sh
echo \"$cmd\"
eval \"$cmd\"
"

fi

done

