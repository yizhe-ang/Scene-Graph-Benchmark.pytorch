N_GPUS=2

srun -p dsta --mpi=pmi2 --gres=gpu:${N_GPUS} -n1 --ntasks-per-node=${N_GPUS} --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
python \
-m torch.distributed.launch \
--nproc_per_node=${N_GPUS} \
tools/relation_train_net_debug.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 2 \
DTYPE "float16" \
SOLVER.MAX_ITER 16000 \
SOLVER.VAL_PERIOD 2000 \
SOLVER.CHECKPOINT_PERIOD 2000 \
GLOVE_DIR artifacts/glove \
MODEL.PRETRAINED_DETECTOR_CKPT artifacts/checkpoints/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR artifacts/checkpoints/psg-transformer-sgdet-exmp \
SOLVER.BASE_LR 0.001 \
SOLVER.SCHEDULE.TYPE WarmupMultiStepLR


# srun -p dsta --mpi=pmi2 --gres=gpu:${N_GPUS} -n1 --ntasks-per-node=${N_GPUS} --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
# python \
# -m torch.distributed.launch \
# --nproc_per_node=${N_GPUS} \
# tools/relation_train_net.py \
# --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
# MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
# MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
# MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
# SOLVER.IMS_PER_BATCH 12 \
# TEST.IMS_PER_BATCH 2 \
# DTYPE "float16" \
# SOLVER.MAX_ITER 50000 \
# SOLVER.VAL_PERIOD 2000 \
# SOLVER.CHECKPOINT_PERIOD 2000 \
# GLOVE_DIR artifacts/glove \
# MODEL.PRETRAINED_DETECTOR_CKPT artifacts/checkpoints/pretrained_faster_rcnn/model_final.pth \
# OUTPUT_DIR artifacts/checkpoints/transformer-sgdet-exmp

# python \
# -m torch.distributed.launch \
# --master_port 10026 \
# --nproc_per_node=2 \
# tools/relation_train_net.py \
# --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
# MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
# MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
# MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
# MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
# MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
# MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
# SOLVER.IMS_PER_BATCH 12 \
# TEST.IMS_PER_BATCH 2 \
# DTYPE "float16" \
# SOLVER.MAX_ITER 50000 \
# SOLVER.VAL_PERIOD 2000 \
# SOLVER.CHECKPOINT_PERIOD 2000 \
# GLOVE_DIR /home/kaihua/glove \
# MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth \
# OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
