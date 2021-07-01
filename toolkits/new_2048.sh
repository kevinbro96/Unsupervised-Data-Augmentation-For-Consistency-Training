python -u tools/main_s1_d.py \
--save_dir results/4loss \
--dim 2048 \
--fdim 32 \
--kl 1.0 \
--ce 1.0 \
--re 1.0 \
--norm 5.0 \
--sp 0.01 0.1 1 \
--optim 'consine'
