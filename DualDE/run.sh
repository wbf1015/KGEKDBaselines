bash myrun.sh 1024 256 6.0 0.005 10000 512 32 0048 models/DualDE_FB15k-237_0042/embedding_checkpoint FB15k-237 1 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
