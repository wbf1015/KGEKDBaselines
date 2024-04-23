bash myrun.sh 1024 256 4.0 0.001 30000 512 32 0058 models/DualDE_FB15k-237_0051/embedding_checkpoint FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 30000

