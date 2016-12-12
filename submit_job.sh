#!/bin/bash
JOB_NAME=$1
DATASET=$2
MI_WT=$3
IMG_DIR=$4
CHECKPOINT_DIR=$5

echo $JOB_NAME
echo $DATASET
echo $MI_WT
echo $IMG_DIR
echo $CHECKPOINT_DIR

qsub -N $JOB_NAME -v dataset=$DATASET,mi_wt=$MI_WT,img_dir=$IMG_DIR,cp_dir=$CHECKPOINT_DIR -l nodes=1:ppn=2:gpus=1:titan,walltime=48:00:00,pmem=8GB -m ae -M sv1358@nyu.edu run_job.sh
