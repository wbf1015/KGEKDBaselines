bash myrun.sh 1024 128 6.0 0.0005 30000 64 32 0064 models/IterDE_wn18rr_0051/embedding_checkpoint wn18rr 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 30000 \
    -soft_loss_weight 0.001 \
    -test_per_steps 5000

bash myrun.sh 1024 128 6.0 0.0002 30000 64 32 0065 models/IterDE_wn18rr_0051/embedding_checkpoint wn18rr 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 30000 \
    -soft_loss_weight 0.001 \
    -test_per_steps 5000

bash myrun.sh 1024 128 8.0 0.0005 30000 64 32 0066 models/IterDE_wn18rr_0051/embedding_checkpoint wn18rr 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 30000 \
    -soft_loss_weight 0.001 \
    -test_per_steps 5000

bash myrun.sh 1024 128 8.0 0.0002 30000 64 32 0067 models/IterDE_wn18rr_0051/embedding_checkpoint wn18rr 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 30000 \
    -soft_loss_weight 0.001 \
    -test_per_steps 5000

