#!/bin/bash
JOB_NAME=$1
DATASET=$2

echo $JOB_NAME
echo $DATASET

qsub -N $JOB_NAME -v dataset=$DATASET -l nodes=1:ppn=2:gpus=1:titan,walltime=55:00:00,pmem=8GB -m ae -M sv1358@nyu.edu run_job.sh
