
# nohup sh run.sh > logs/train_VOC_15-5_use_KD_layer_weight.log 2>&1 &
## VOC
DATA_ROOT=/home/yb/dataset/VOC/PascalVOC12/VOCdevkit/VOC2012
DATASET=voc
TASK=15-1
EPOCH=30
BATCH=12
LOSS=bce_loss
KD_LOSS=KD_loss
LR=0.01
THRESH=0.7
MEMORY=0
CKPT=checkpoints/
DATA_RATIO=1.0

python run.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 1,0 --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} --KD_loss_type ${KD_LOSS} --use_KD_layer_weight\
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --unknown --w_transfer --amp --mem_size ${MEMORY} \
    --feature_decoupling --data_ratio ${DATA_RATIO} 



## nohup sh run.sh > logs/train_ADE_100-10.log 2>&1 &
## ADE20K
# DATA_ROOT=/home/yb/dataset/ADEChallengeData2016/
# DATASET=ade
# TASK=100-10 # [100-5, 100-10, 100-50, 50-50]
# EPOCH=30
# BATCH=12
# LOSS=bce_loss
# KD_LOSS=KD_loss
# LR=0.01
# THRESH=0.7
# MEMORY=0 # [0 (for SSUL), 300 (for SSUL-M)]
# CKPT=checkpoints/
# DATA_RATIO=1.0

# python run.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 3,2,1,0 \
#     --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} --KD_loss_type ${KD_LOSS} --use_KD_layer_weight\
#     --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly --pseudo --pseudo_thresh ${THRESH} \
#     --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY} \
#     --feature_decoupling --data_ratio ${DATA_RATIO} 




## nohup sh run.sh > logs/train_ISPRS_4-1_KDoutlogits.log 2>&1 &
## ISPRS
# DATA_ROOT=/home/yb/dataset/ISPRS_2D/postdam/Incremental_RGB_600
# DATASET=ISPRS
# TASK=2-2-1 #[4-1, 2-3, 2-2-1, 2-1, offline]
# EPOCH=30
# BATCH=12
# LOSS=bce_loss
# KD_LOSS=KD_loss
# LR=0.01
# THRESH=0.7
# MEMORY=0
# CKPT=checkpoints/

# python run.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --lr ${LR} \
#     --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} --KD_loss_type ${KD_LOSS} --use_KD_layer_weight\
#     --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
#     --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
#     --unknown --w_transfer --amp --mem_size ${MEMORY} 