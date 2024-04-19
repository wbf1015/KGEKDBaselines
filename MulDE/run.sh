bash myrun.sh 1024 256 6.0 0.005 10000 512 32 0050 0.001 FB15k-237 1 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -pretrain_path1 models/TransE_128_FB15k-237_0/checkpoint \
    -pretrain_path2 models/TransE_128_FB15k-237_1/checkpoint \
    -pretrain_path3 models/TransE_128_FB15k-237_2/checkpoint \
    -pretrain_path4 models/TransE_128_FB15k-237_3/checkpoint \
    -soft_loss_weight 0.01 -k 100 -teacher_num 4 \
    -weight_decay 0.000
