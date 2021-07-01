cd /gdata2/yangkw/auto_aug-master;
CUDA_LAUNCH_BLOCKING=1;
python -u tools/main_s1_nn.py \
--save_dir ./results/emb128 \
--dim 128

