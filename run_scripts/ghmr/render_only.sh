cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung

# JAMES_RESULT=/home/groups/syyeung/jmhb/bio-pose/mdm/save/
# pretrained_model=$JAMES_RESULT/20230304_trainemp_exp19-0-amasshml_augment_FcShapeAxyzAvel-amasshml_augment_FcShapeAxyzAvel/000003/model000410000.pt

#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000002/model000600100.pt
#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000003/model000290000.pt
#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000014/model000025000.pt
#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000014/model000045000.pt
#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000137/model000345000.pt
pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000138/model000345000.pt

pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000156/model000380000.pt
#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000177/model000375000.pt

#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000176/model000375000.pt

#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000177/model000375000.pt

#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000186/model000355000.pt

# FINAL - editing
#pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000189/model000400000.pt

# FINAL
pretrained_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000190/model000355000.pt

for data in amasshml_FcShapeAxyzAvel; do 

expname=${cur_fname}-${db}-${data}
cmd="python -m gthmr.render_only \
		--data_config_path gthmr/emp_train/config/data/${data}.yml \
		--save_dir gthmr/results/${expname} \
		--model_path $pretrained_model 
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
#SBATCH --mem=48gb
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
