bash myrun.sh 1024 256 8.0 0.0002 10000 512 256 0023 models/RotatE_wn18rr_1/checkpoint wn18rr 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -soft_loss_weight 0.1 \
