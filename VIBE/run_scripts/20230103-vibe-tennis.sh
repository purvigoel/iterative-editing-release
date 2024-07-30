cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung

for vid_key in  carlos \
                casper \
                college-baltazar \
                dominic \
                emma \
                felix \
                college-gabe \
                jannik \
                novak \
                stan \
                college-laura \
; do

cmd="python -m scripts.20230103_run_vibe_tennis.py \
    --vid_key=${vid_key}
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