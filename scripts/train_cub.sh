. CONFIG

ID=1
GPU=1
NC=4
CLS=0.5
INT=1.0
NGF=128
NDF=64
MI_WT=1

if [ -n "$1" ]
then
    MI_WT=$1
fi

if [ -n "$2" ]
then
    CUB_IMG_OUT_DIR=$2
fi

if [ -n "$3" ]
then
    CHECKPOINT_DIR=$3
fi

echo $MI_WT
echo $CUB_IMG_DIR
echo $CHECKPOINT_DIR

display_id=10${ID} \
gpu=${ID} \
dataset="cub" \
name="cub_v2_nc${NC}_cls${CLS}_int${INT}_ngf${NGF}_ndf${NDF}" \
cls_weight=${CLS} \
interp_weight=${INT} \
interp_type=1 \
mi_weight=$MI_WT \
niter=600 \
nz=100 \
lr_decay=0.5 \
decay_every=100 \
img_dir=${CUB_IMG_DIR} \
data_root=${CUB_META_DIR} \
classnames=${CUB_META_DIR}/allclasses.txt \
trainids=${CUB_META_DIR}/trainvalids.txt \
init_t=${CUB_NET_TXT} \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
out_image_dir=${CUB_IMG_OUT_DIR} \
numCaption=${NC} \
print_every=4 \
save_every=1 \
replicate=0 \
use_cudnn=1 \
ngf=${NGF} \
ndf=${NDF} \
th main_cls_int.lua

