import os
import sys

from argparse import ArgumentParser
import logging 
# from torch.utils.tensorboard import SummaryWriter

import torch

import data_loader
from models import EfficientNetB5
from trainer import training_epoch, validation_epoch

def main():
    # Experiment Configuration
    parser = training_specific_args()
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    logging.getLogger().setLevel(logging.INFO)

    # GPU setup    
    use_cuda = config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f'using device: {device}')
    config.device = device # Adding device to config, not already in argparse
    config.num_workers = 1 if use_cuda else 4 # for dataloader
    config.pin_memory = False if use_cuda else False

    # Data loading
    train_loader = data_loader.train_dataloader(config)
    val_loader = data_loader.val_dataloader(config)
    # test_loader = data_loader.test_dataloader(config)
    
    # Model
    model = EfficientNetB5(config.pretrained)
    model.to(device)

    # Optims - Scheduler monitors recall metric, hence mode = max
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           patience=5,
                                                           factor=0.3, verbose=True)

    if config.resume_pt:
        logging.info(f'Loading {config.resume_pt}')
        state = torch.load(f'../checkpoints/{config.resume_pt}')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

    logging.info('Start training procedure')
    val_loss_min = float('inf')

    for epoch in range(1, config.epochs+1):
        training_epoch(config, model, train_loader, optimizer, epoch)
        val_loss = validation_epoch(config, model, val_loader)
        scheduler.step(val_loss)
        
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            state = {
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, f'{config.save_dir}/{config.exp_name}_{val_loss}.pt')

def training_specific_args():  

        parser = ArgumentParser()

        # GPU
        parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--seed', default=400, type=int)

        # data
        parser.add_argument('--data_root', default=f'{os.path.dirname(os.getcwd())}/input', type=str)
        parser.add_argument('--val_folds', default= 4, type=int)
        parser.add_argument('--test_id', default=0, choices=range(4), type=int, help='parquet file id') 

        # network params
        parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        
        # training params
        parser.add_argument('--epochs', default=2, type=int)
        parser.add_argument('--batch_size', default=4, type=int)
        
        parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--resume_pt', type=str)

        # output
        parser.add_argument('--save_dir', default=f'{os.path.dirname(os.getcwd())}/checkpoints', type=str)
        parser.add_argument('--exp_name', default=f'run_1', type=str)
        parser.add_argument('--log_interval', type=int, default=1,
                            help='how many batches to wait before logging training status')        
        return parser

if __name__=="__main__":
    main()