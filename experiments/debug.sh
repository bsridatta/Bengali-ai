python 3 ../src/train.py
--cuda True\ 
--seed 400\ 
--data_root ../input\ 
--train_folds [0, 1, 2]\ 
--val_folds [3]\ 
--test_id 0\ 
--pretrained False\ 
--learning_rate 0.0001\ 
--epochs 2\ 
--batch_size 4\ 
--fast_dev_run True\ 
--save_dir ../checkpoints\ 
--exp_name debug\ 
--log_interval 1\ 

sudo shutdown now