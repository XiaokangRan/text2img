#!/bin/bash
module purge
module load cuda/7.5.18
module load cudnn/7.5v5.1
module load jpeg/intel/9a

cd /home/sv1358/text2img

if [ "$dataset" == "cub" ]
then
    ./scripts/train_cub.sh $mi_wt $img_dir $cp_dir
elif [ "$dataset" == "flowers" ]
then
    ./scripts/train_flowers.sh $mi_wt $img_dir $cp_dir
elif [ "$dataset" == "coco" ]
then
    ./scripts/train_coco.sh
else
    echo "Please enter a valid dataset"
fi
