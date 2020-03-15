python3.7 ../src/train.py \
    --cuda True\
    --seed 400\
    --data_root ../input\
    --val_folds 4\
    --test_id 0\
    --cutmix 0.5\
    --pretrained True\
    --freeze_blocks 20\
    --learning_rate 0.0001\
    --epochs 100\
    --batch_size 48\
    --fast_dev_run False\
    --save_dir ../checkpoints\
    --exp_name b5100e\
    --log_interval 1

# sudo shutdown now