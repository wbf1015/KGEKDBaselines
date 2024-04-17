bash myrun.sh 1024 256 4.0 0.0002 10000 512 256 0016 models/TransE_512_FB15k-237_1 FB15k-237 1 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -soft_loss_weight 0.1


