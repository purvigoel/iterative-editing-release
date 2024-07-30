
for n in '1985' '1986' '1987' '1988' '1989' '1990' '1991' '1992' '1993' '1994' '1995' '1996' '1997' '1998' '1999' '2000' '2001' '2002' '2003' '2004' '2005' '2006' '2007' '2008' '2009' '2010' '2011' '2012' '2013' '2014' '2015' '2016' '2017' '2018' '2019' '2020' '2021' '2022' '2023' '2024' '2025' '2026' '2027' '2028' '2029' '2030' '2031' '2032' '2033'; do 

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