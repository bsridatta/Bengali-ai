import torch.nn.functional as F
from collections import OrderedDict
import torch

def training_epoch(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device) if key != 'image_id' else batch[key]
        optimizer.zero_grad()
        output = _training_step(batch, batch_idx, model)
        loss = output['loss_val']
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))   


def validation_epoch(args, model, val_loader):
    model.eval()
    loss = 0
    acc = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(args.device) if key != 'image_id' else batch[key]
            output = _validation_step(batch, batch_idx, model)
            loss += output['val_loss'].item()
            acc += output['val_acc']
    avg_loss = loss/len(val_loader)
    avg_acc = acc/len(val_loader)
    print(f'Val set: Average Loss: {avg_loss}, Accuracy: {avg_acc}')

    return avg_loss

def _training_step(batch, batch_idx, model):

    grapheme, vowel, consonant = model(batch['image'])

    loss_grapheme = F.cross_entropy(grapheme, batch['grapheme_root'].long())
    loss_vowel = F.cross_entropy(vowel, batch['vowel_diacritic'].long())
    loss_consonant = F.cross_entropy(consonant, batch['consonant_diacritic'].long())
    loss_val = loss_grapheme + loss_vowel + loss_consonant
    
    logger_logs = {
        "TLoss_G": loss_grapheme, 
        "TLoss_V": loss_vowel, 
        "TLoss_C": loss_consonant
    }

    return OrderedDict({'loss_val': loss_val, 'log': logger_logs})

def _validation_step(batch, batch_idx, model):
        grapheme, vowel, consonant = model(batch['image'])

        loss_grapheme = F.cross_entropy(grapheme, batch['grapheme_root'].long())
        loss_vowel = F.cross_entropy(vowel, batch['vowel_diacritic'].long())
        loss_consonant = F.cross_entropy(consonant, batch['consonant_diacritic'].long())
        loss_val = loss_grapheme + loss_vowel + loss_consonant
        
        acc_grapheme = torch.sum(grapheme.argmax(dim=1) == batch["grapheme_root"]).item() / (len(grapheme) * 1.0)
        acc_vowel = torch.sum(vowel.argmax(dim=1) == batch["vowel_diacritic"]).item() / (len(vowel) * 1.0)
        acc_consonant = torch.sum(consonant.argmax(dim=1) == batch["consonant_diacritic"]).item() / (len(consonant) * 1.0)
        val_acc = acc_grapheme + acc_vowel + acc_consonant

        logger_logs = {"VLoss_G": loss_grapheme, 
                       "VLoss_V": loss_vowel, 
                       "VLoss_C": loss_consonant,
                       "VAcc_G": acc_grapheme, 
                       "VAcc_V": acc_vowel, 
                       "VAcc_C": acc_consonant,
                       "VAcc": val_acc,
                       "Vloss": loss_val
                    }

        return OrderedDict({'val_loss': loss_val, 'val_acc': val_acc, 'log': logger_logs})
    