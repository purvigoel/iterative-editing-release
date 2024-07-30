
for n in '0789' '0790' '0791' '0792' '0793' '0794' '0795' '0796' '0797' '0798' '0799' '0800' '0801' '0802' '0803' '0804' '0805' '0806' '0807' '0808' '0809' '0810' '0811' '0812' '0813' '0814' '0815' '0816' '0817' '0818' '0819' '0820' '0821' '0822' '0823' '0824' '0825' '0826' '0827' '0828' '0829' '0830' '0831' '0832' '0833' '0834' '0835' '0836' '0837' '0838' '0168' '0169' '0170' '0171' '0172' '0173' '0174' '0175' '0176' '0177' '0178' '0179' '0180' '0181' '0182' '0183' '0184' '0185'; do 


cmd="python demo2.py \
	--input_image_folder=/oak/stanford/groups/syyeung/wangkua1/data/data/Penn_Action/frames/${n} \
	--output_folder=/oak/stanford/groups/syyeung/wangkua1/data/data/Penn_Action/vs_results \
	--detector maskrcnn \
	--render 0 \
	--tracking_method pose \
	--run_smplify \
	--no_render
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