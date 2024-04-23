bash myrun.sh 1024 256 6.0 0.0005 10000 512 32 0051 0.000 FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -pretrain_path1 models/RotatE_FB15k-237_10/checkpoint \
    -pretrain_path2 models/RotatE_FB15k-237_11/checkpoint \
    -pretrain_path3 models/RotatE_FB15k-237_12/checkpoint \
    -pretrain_path4 models/RotatE_FB15k-237_13/checkpoint \
    -soft_loss_weight 0.01 -k 100 -teacher_num 4



bash myrun.sh 1024 256 4.0 0.005 10000 512 32 0052 0.000 FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -pretrain_path1 models/RotatE_FB15k-237_10/checkpoint \
    -pretrain_path2 models/RotatE_FB15k-237_11/checkpoint \
    -pretrain_path3 models/RotatE_FB15k-237_12/checkpoint \
    -pretrain_path4 models/RotatE_FB15k-237_13/checkpoint \
    -soft_loss_weight 0.01 -k 100 -teacher_num 4

bash myrun.sh 1024 256 4.0 0.0005 10000 512 32 0053 0.000 FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -pretrain_path1 models/RotatE_FB15k-237_10/checkpoint \
    -pretrain_path2 models/RotatE_FB15k-237_11/checkpoint \
    -pretrain_path3 models/RotatE_FB15k-237_12/checkpoint \
    -pretrain_path4 models/RotatE_FB15k-237_13/checkpoint \
    -soft_loss_weight 0.01 -k 100 -teacher_num 4



bash myrun.sh 1024 256 8.0 0.005 10000 512 32 0054 0.000 FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -pretrain_path1 models/RotatE_FB15k-237_10/checkpoint \
    -pretrain_path2 models/RotatE_FB15k-237_11/checkpoint \
    -pretrain_path3 models/RotatE_FB15k-237_12/checkpoint \
    -pretrain_path4 models/RotatE_FB15k-237_13/checkpoint \
    -soft_loss_weight 0.01 -k 100 -teacher_num 4

bash myrun.sh 1024 256 8.0 0.0005 10000 512 32 0055 0.000 FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -pretrain_path1 models/RotatE_FB15k-237_10/checkpoint \
    -pretrain_path2 models/RotatE_FB15k-237_11/checkpoint \
    -pretrain_path3 models/RotatE_FB15k-237_12/checkpoint \
    -pretrain_path4 models/RotatE_FB15k-237_13/checkpoint \
    -soft_loss_weight 0.01 -k 100 -teacher_num 4