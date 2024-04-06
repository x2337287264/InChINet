from Dataset import MyDataset, get_data_loader
from MyTransformer import MyTransformer
from nt_xent_loss import NTXentLoss
import yaml
from torch import optim
import torch
import os
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random


def seed_all(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pretrain():
    seed = 3407
    seed_all(seed)
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    pretrain_dataset = MyDataset(config['processed_smile_directory'], config['tokenized_inchi_directory'], config['smile_vocab_path'], 
                                 config['inchi_vocab_path'], config['smile_vocab_bpe_path'], config['smile_codes_path'], config['max_seq_len'])
    train_loader, valid_loader = get_data_loader(pretrain_dataset, config['valid_ratio'], config['batch_size'], config['num_workers'])
    print(len(train_loader), len(valid_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyTransformer(config['batch_size'], config['vocab_size'], config['d_model'], config['max_seq_len'], config['project_dimension']).to(device)
    # model.load_state_dict(torch.load('valid_model/model_epoch2_iter-1.pkl'))
    nt_xent_loss = NTXentLoss(device, config['batch_size'], config['temperature'], True)
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=1e-4)
    # scheduler = ExponentialLR(optimizer, gamma=0.8)
    # best_valid_loss = 10000
    loss_sum = 0
    for i in range(config['epochs']):
        for j, (smile_aug, inchi_aug) in enumerate(train_loader):
            optimizer.zero_grad()
            smile_aug = smile_aug.to(device)
            inchi_aug = inchi_aug.to(device)

            output1 = model(smile_aug)  # (B,T)
            output2 = model(inchi_aug)  # (B,T)
            loss = nt_xent_loss(output1.reshape(config['batch_size'], -1), output2.reshape(config['batch_size'], -1))
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if (j+1) % 100 == 0:
                loss_sum = loss_sum/100
                print('Training... epoch: {} , iter: {} , loss: {} '.format(i, j+1, loss_sum))
                loss_sum = 0
            if (j+1) % 100000 == 0:
                loss = evaluate(model, nt_xent_loss, valid_loader, device, config)
                print('Validating... epoch: {} , iter: {} , loss: {} '.format(i, j+1, loss))
                print("saving model...")
                if not os.path.exists('valid_model'):
                    os.makedirs('valid_model')
                torch.save(model.state_dict(), 'valid_model/model_epoch{}_iter{}.pkl'.format(i, j+1))
        torch.save(model.state_dict(), 'valid_model/model_epoch{}_iter{}.pkl'.format(i, -1))
        

def evaluate(model, nt_xent_loss, valid_loader, device, config):
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        count = 0
        for smile_aug, inchi_aug in valid_loader:
            smile_aug = smile_aug.to(device)
            inchi_aug = inchi_aug.to(device)
            output1 = model(smile_aug)
            output2 = model(inchi_aug)

            loss = nt_xent_loss(output1.reshape(config['batch_size'], -1), output2.reshape(config['batch_size'], -1))
            valid_loss += loss.item()
            count += 1
        valid_loss /= count
    model.train()
    return valid_loss


pretrain()
