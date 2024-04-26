bash myrun.sh 128 1024 8.0 0.0002 30000 512 32 0075 models/DualDE_wn18rr_0067/embedding_checkpoint wn18rr 1 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 30000 \
    -test_per_steps 2500 \

