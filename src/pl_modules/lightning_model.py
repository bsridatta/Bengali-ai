import logging as log
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

import pytorch_lightning
from pytorch_lightning.core import LightningModule
from pytorch_lightning.core import data_loader

from efficientnet_pytorch import EfficientNet

import sys
sys.path.append("../src/")
from dataset import BengaliAI

class EfficientNet(LightningModule):


    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(EfficientNet, self).__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(3, 137, 236)

        # build model
        self.__build_model()



    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Model Components
        """
        backbone_model = EfficientNet.from_pretrained('efficientnet-b5')
        # Take the whole resnext except for the last layer
        backbone_layers = torch.nn.ModuleList(backbone_model.children())[:-2]
        # Unpack all layers to Sequential as list is not a valid parameter 
        self.features = torch.nn.Sequential(*backbone_layers)
        in_features = backbone_model._fc.in_features
        self.fc_grapheme_root = torch.nn.Linear(in_features, 168)
        self.fc_vowel_diacritic = torch.nn.Linear(in_features, 11)
        self.fc_consonant_diacritic = torch.nn.Linear(in_features, 7)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        grapheme = self.fc_grapheme_root(x)
        vowel = self.fc_vowel_diacritic(x)
        consonant = self.fc_consonant_diacritic(x) 

        return grapheme, vowel, consonant

    def training_step(self, batch, batch_idx):
        grapheme, vowel, consonant = self.forward(batch['image'])

        loss_grapheme = F.cross_entropy(grapheme, batch['grapheme_root'].long())
        loss_vowel = F.cross_entropy(vowel, batch['vowel_diacritic'].long())
        loss_consonant = F.cross_entropy(consonant, batch['consonant_diacritic'].long())
        loss_val = loss_grapheme + loss_vowel + loss_consonant
        
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # TODO is this needed? We dont really since we only use 1 GPU!
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        
        logger_logs = {"TLoss_G": loss_grapheme, 
                       "TLoss_V": loss_vowel, 
                       "TLoss_C": loss_consonant
                    }

        return OrderedDict({'loss_val': loss_val, 'log': logger_logs})

    def validation_step(self, batch, batch_idx):
        grapheme, vowel, consonant = self.forward(batch['image'])

        loss_grapheme = F.cross_entropy(grapheme, batch['grapheme_root'].long())
        loss_vowel = F.cross_entropy(vowel, batch['vowel_diacritic'].long())
        loss_consonant = F.cross_entropy(consonant, batch['consonant_diacritic'].long())
        loss_val = loss_grapheme + loss_vowel + loss_consonant
        
        acc_grapheme = torch.sum(grapheme == batch["grapheme_root"]).item() / (len(grapheme) * 1.0)
        acc_vowel = torch.sum(grapheme == batch["vowel_diacritic"]).item() / (len(vowel) * 1.0)
        acc_consonant = torch.sum(grapheme == batch["consonant_diacritic"]).item() / (len(consonant) * 1.0)
        val_acc = acc_grapheme + acc_vowel + acc_consonant
        val_acc = torch.tensor(val_acc) # TODO Why this? maybe used in monitoring hence it needs to be a scalar tensor?

        # TODO Why?
        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        # TODO can have val_acc in logs as well, is val_acc also a 'keyword' output?
        logger_logs = {"VLoss_G": loss_grapheme, 
                       "VLoss_V": loss_vowel, 
                       "VLoss_C": loss_consonant,
                       "VAcc_G": acc_grapheme, 
                       "VAcc_V": acc_vowel, 
                       "VAcc_C": acc_consonant
                    }

        return OrderedDict({'val_loss': loss_val, 'val_acc': val_acc, 'log': logger_logs})
        
    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # TODO-Whats happening Multi GPU? - reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # TODO-Whats happening - reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}

        logger_logs = {"aVLoss_G": torch.stack([x['log']['VLoss_G'] for x in outputs]).mean(), 
                       "aVLoss_V": torch.stack([x['log']['VLoss_V'] for x in outputs]).mean(), 
                       "aVLoss_C": torch.stack([x['log']['VLoss_C'] for x in outputs]).mean(), 
                       "aVAcc_G": torch.stack([x['log']['VAcc_G'] for x in outputs]).mean(), 
                       "aVAcc_V": torch.stack([x['log']['VAcc_V'] for x in outputs]).mean(), 
                       "aVAcc_C": torch.stack([x['log']['VAcc_C'] for x in outputs]).mean(),
                       "Mean_Val_Loss": val_loss_mean                 
                    }

        result = {'progress_bar': tqdm_dict, 'log': logger_logs, 'val_loss': val_loss_mean}
        
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         patience=5,
                                                         factor=0.3, verbose=True)
        return [optimizer], [scheduler]

    def __dataloader(self, train=True):

        batch_size = self.hparams.batch_size

        folds =  self.hparams.folds
        test_id = self.hparams.test_id
        data_root = self.hparams.data_root
        dataset = BengaliAI(folds=folds, train=train, 
                            test_id=test_id, data_root=data_root)
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
        )

        return loader

    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=True)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  

        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--learning_rate', default=1e-4, type=float)

        # data
        parser.add_argument('--data_root', default=f'{os.path.dirname(os.getcwd())}/input', type=str)
        parser.add_argument('--train_folds', default=[0,1,2,3], type=list)
        parser.add_argument('--val_folds', default=[4], type=list)
        parser.add_argument('--test_id', default=0, type=int)

        # training params (opt)
        parser.add_argument('--batch_size', default=64, type=int)

        return parser