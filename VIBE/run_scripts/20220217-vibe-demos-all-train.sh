


# for sn in  \
# 	'S5' 'S6' 'S7' 'S8' \
# 	; do


# for action in  \
# 	"Directions 1" "Directions" "Discussion 1" "Discussion" "Greeting 1" "Greeting" "Posing 1" "Posing" "Purchases 1" "Purchases" "SittingDown 2" "SittingDown" "TakingPhoto 1" "TakingPhoto" "Waiting 1" "Waiting" "Walking 1" "Walking" "WalkingDog 1" "WalkingDog" "WalkTogether 1" "WalkTogether" \
# 	; do


# for camera in \
# 	"54138969" "55011271" "58860488" "60457274" \
# 	; do 

# action=$(printf %q "$action")
# fname="${action}.${camera}.mp4"

# cmd="python demo.py \
# 	--vid_file=/scratch/groups/syyeung/hmr_datasets/h36m_train/${sn}/Videos/${fname} \
# 	--output_folder=/scratch/users/wangkua1/results/vibe/${sn}/${fname}
# "


for sn in  \
	'S5' 'S6' 'S7' 'S8' \
	; do


for vid_id in  {9..128}; do

fname=$(ls /scratch/groups/syyeung/hmr_datasets/h36m_train/${sn}/Videos/  | sed -n ${vid_id}p)
fname=$(printf %q "$fname") # Escapae whitespaces

cmd="python demo.py \
	--vid_file=/scratch/groups/syyeung/hmr_datasets/h36m_train/${sn}/Videos/${fname} \
	--output_folder=/scratch/users/wangkua1/results/vibe/${sn}/${fname}
"


if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=vibe-demos5
#SBATCH --output=slurm_logs/vibe-demos5-%j-out.txt
#SBATCH --error=slurm_logs/vibe-demos5-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 1
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
done
done

