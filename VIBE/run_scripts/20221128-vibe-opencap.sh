cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung

for action in   armROM \
                balance \
                balanceBigger \
                DJ \
                neutral \
                run \
                SLDJ \
                SLJump \
                SLS \
                sprint \
                sprint2 \
                squatJump \
                stairs \
                star \
                STS \
                walk \
                walkTurn \
                Y \
; do
for cam_id in 0 1 2; do

R='/home/groups/syyeung/wangkua1/data/OpenCapData_03284efb-2244-4a48-aec9-abc34afdffc8'
cmd="python demo2.py \
    --vid_file=$R/Videos/Cam${cam_id}/InputMedia/${action}/${action}_sync.mp4 \
    --output_folder=/home/groups/syyeung/wangkua1/opencap/exps/${action}_${cam_id} \
    --detector maskrcnn
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
done
done