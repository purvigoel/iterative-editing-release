cur_fname="$(basename $0 .sh)"
partition=gpu,syyeung

# JAMES_RESULT=/home/groups/syyeung/jmhb/bio-pose/mdm/save/
# pretrained_model=$JAMES_RESULT/20230304_trainemp_exp19-0-amasshml_augment_FcShapeAxyzAvel-amasshml_augment_FcShapeAxyzAvel/000003/model000410000.pt

pose_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000014/model000160000.pt
motion_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000015/model000600100.pt
motion_model=/raid/pgoel2/bio-pose/mr_dm_data/pretrained/model000280000.pt

motion_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000047/model000360000.pt


motion_model=/raid/pgoel2/bio-pose/mdm/save/20230306_trainemp_exp29-0-amasshml_FcShapeAxyz-amasshml_FcShapeAxyz/000059/model000585000.pt


for data in amasshml_FcShapeAxyzAvel; do 

expname=${cur_fname}-${db}-${data}

seed=1
num_samples=100
savedir="test4/0_100_1/"
rollout=4

indir="diffuseIK_itr0.npy"

printf "seed: %s\n" "${seed}"
printf "savedir: %s\n" "${savedir}"


cmd4="python -m generative_infill.generative_infill \
                --seed ${seed}
                --filename $indir
                --outname diffuseIK_itr${rollout}.npy
                --itr 2
                --blend 0.8
                --num_samples ${num_samples}
"


echo $cmd4
eval $cmd4
echo "Elapsed Time interpolate: $SECONDS"

done
