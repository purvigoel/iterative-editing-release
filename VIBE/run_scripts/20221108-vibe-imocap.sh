cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung

for action in   ballet \
                ballet2 \
                bulgarian_split_squat \
                golf \
                jump \
                pitching \
                pushup1 \
                rafael \
                serve \
                shotput \
                situp \
                sprinting2 \
                swing \
                taiji \
                tiger_woods_golf \
                weight_lifting_mirror \
                weight_lifting2 \
                yoga_2 \
                yoga_3_mirror \
                yoga_dog \
; do

cmd="python -m scripts.20221108_run_vibe_imocap \
    --action $action \
"

if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
# break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${cur_fname}-${partition}
#SBATCH --output=slurm_logs/${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/${cur_fname}-${partition}-%j-err.txt
#SBATCH --mem=32gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p ${partition}
#SBATCH --time=48:00:00

#necessary env
# source /home/users/wangkua1/setup_rl.sh
echo \"$cmd\"
eval \"$cmd\"
"

fi

done
done