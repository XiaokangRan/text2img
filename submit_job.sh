#!/bin/bash
JOB_NAME=$1
DATASET=$2
MI_WT=$3
IMG_DIR=$4
CHECKPOINT_DIR=$5
USE_TEXT=false
if [ $# -gt 5 ]
then
    USE_TEXT=true
fi

echo $JOB_NAME
echo $DATASET
echo $MI_WT
echo $IMG_DIR
echo $CHECKPOINT_DIR
echo $USE_TEXT

qsub -N $JOB_NAME -v dataset=$DATASET,mi_wt=$MI_WT,img_dir=$IMG_DIR,cp_dir=$CHECKPOINT_DIR,use_text=$USE_TEXT -l nodes=1:ppn=2:gpus=1:titan,walltime=48:00:00,pmem=8GB -m ae -M sv1358@nyu.edu run_job.sh
