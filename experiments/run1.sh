python 3 ../src/train.py
--cuda True\ 
--seed 400\ 
--data_root ../input\ 
--train_folds [0, 1, 2]\ 
--val_folds [3]\ 
--test_id 0\ 
--pretrained True\ 
--learning_rate 0.0001\ 
--epochs 100\ 
--batch_size 64\ 
--fast_dev_run False\ 
--save_dir ../checkpoints\ 
--exp_name run_1\ 
--log_interval 1\ 

sudo shutdown now