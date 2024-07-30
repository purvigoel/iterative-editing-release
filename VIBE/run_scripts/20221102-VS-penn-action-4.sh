
for n in '2146' '2147' '2148' '2149' '2150' '2151' '2152' '2153' '2154' '2155' '2156' '2157' '2158' '2159' '2160' '2161' '2162' '2163' '2164' '2165' '2166' '2167' '2168' '2169' '2170' '2171' '2172' '2173' '2174' '2175' '2176' '2177' '2178' '2179' '2180' '2181' '2182' '2183' '2184' '2185' '2186' '2187' '2188' '2189' '2190' '1984' ; do 


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