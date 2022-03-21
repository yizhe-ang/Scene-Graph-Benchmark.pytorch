N_GPUS=1
NODE=10028

# python \
# -m torch.distributed.launch \
# --master_port 10027 \
# --nproc_per_node=1 \
# tools/relation_test_net.py \
# --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
# MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
# MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
# MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
# TEST.IMS_PER_BATCH 1 \
# DTYPE "float16" \
# GLOVE_DIR /home/kaihua/glove \
# MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp \
# OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp

srun -p dsta --mpi=pmi2 --gres=gpu:${N_GPUS} -n1 --ntasks-per-node=${N_GPUS} --job-name=job --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
python \
-m torch.distributed.launch \
--master_port ${NODE} \
--nproc_per_node=${N_GPUS} \
tools/relation_test_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
TEST.IMS_PER_BATCH 1 \
DTYPE "float16" \
GLOVE_DIR artifacts/glove \
MODEL.PRETRAINED_DETECTOR_CKPT artifacts/checkpoints/causal-motif-sgdet-exmp \
OUTPUT_DIR artifacts/checkpoints/causal-motif-sgdet-exmp
