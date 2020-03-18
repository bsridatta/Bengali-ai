python3.7 ../src/train.py \
    --cuda True\
    --seed 400\
    --data_root ../input\
    --val_folds 3\
    --test_id 0\
    --cutmix 0\
    --pretrained True\
    --freeze_blocks 20\
    --learning_rate 0.0001\
    --epochs 15\
    --batch_size 512\
    --fast_dev_run False\
    --save_dir ../checkpoints\
    --exp_name fold_3_\
    --log_interval 1


python3.7 ../src/train.py \
    --cuda True\
    --seed 400\
    --data_root ../input\
    --val_folds 2\
    --test_id 0\
    --cutmix 0\
    --pretrained True\
    --freeze_blocks 20\
    --learning_rate 0.0001\
    --epochs 15\
    --batch_size 512\
    --fast_dev_run False\
    --save_dir ../checkpoints\
    --exp_name fold_2_\
    --log_interval 1

python3.7 ../src/train.py \
    --cuda True\
    --seed 400\
    --data_root ../input\
    --val_folds 1\
    --test_id 0\
    --cutmix 0\
    --pretrained True\
    --freeze_blocks 20\
    --learning_rate 0.0001\
    --epochs 15\
    --batch_size 512\
    --fast_dev_run False\
    --save_dir ../checkpoints\
    --exp_name fold_1_\
    --log_interval 1


python3.7 ../src/train.py \
    --cuda True\
    --seed 400\
    --data_root ../input\
    --val_folds 0\
    --test_id 0\
    --cutmix 0\
    --pretrained True\
    --freeze_blocks 20\
    --learning_rate 0.0001\
    --epochs 15\
    --batch_size 512\
    --fast_dev_run False\
    --save_dir ../checkpoints\
    --exp_name fold_0_\
    --log_interval 1

sudo shutdown now