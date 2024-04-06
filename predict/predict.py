from Dataset_predict import MyDataset, get_data_loader
import sys
sys.path.append("..")
from pretrain.nt_xent_loss import NTXentLoss
from pretrain.MyTransformer import MyTransformer
from pretrain.data_process import load_vocab
from MLP import MLP
import yaml
from torch import optim
import torch
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import random
import numpy as np
from torch.optim import lr_scheduler
import pandas as pd

PAD = 0
UNK = 1
BOS = 2
EOS = 3
MASK = 4




def evaluate_transformer(transformer, valid_loader, nt_xent_loss, config, device):
    transformer.eval()
    with torch.no_grad():
        valid_loss = 0
        count = 0
        for j, (_, _, aug1, aug2) in enumerate(valid_loader):
            aug1 = aug1.to(device)
            aug2 = aug2.to(device)
            output1 = transformer(aug1)
            output2 = transformer(aug2)
            loss = nt_xent_loss(output1.reshape(-1, config['max_seq_len']*config['project_dimension']), output2.reshape(-1, config['max_seq_len']*config['project_dimension']))
            valid_loss += loss.item()
            count += 1
        valid_loss /= count
    transformer.train()
    return valid_loss

def finetune_transformer(transformer, train_loader, valid_loader, nt_xent_loss, config, device):
    transformer.train()
    optimizer = optim.Adam(transformer.parameters(), lr=float(config['lr_finetune']))
    best_valid_loss = 10000
    for i in range(config['epochs_finetune']):
        loss_sum = 0
        for j, (_, _, aug1, aug2) in enumerate(train_loader):
            optimizer.zero_grad()
            aug1 = aug1.to(device)
            aug2 = aug2.to(device)
            output1 = transformer(aug1)
            output2 = transformer(aug2)
            loss = nt_xent_loss(output1.reshape(-1, config['max_seq_len']*config['project_dimension']), output2.reshape(-1, config['max_seq_len']*config['project_dimension']))
            loss_sum += loss.item()
            if (j+1) % config['print_every_epochs'] == 0:
                print('Training.. epoch: {} , iter {}, loss is {}'.format(i, j+1, loss_sum/config['print_every_epochs']))
                loss_sum = 0
            if (j+1) % config['print_every_epochs'] == 0:
                valid_loss = evaluate_transformer(transformer, valid_loader, nt_xent_loss, config, device)
                if valid_loss < best_valid_loss:
                    print('saving tranformer... epoch:{}, iter:{}, best_valid_loss:{} '.format(i, j+1, valid_loss))
                    torch.save(transformer.state_dict(), 'procedure_model/transformer_finetune_'+config['dataset']+'.pkl')
                    best_valid_loss = valid_loss
            loss.backward()
            optimizer.step()
    print('finetune transformer complete!')


def evaluate_mlp(mlp, transformer, valid_loader, device):
    mlp.eval()
    transformer.eval()
    # CELoss = torch.nn.CrossEntropyLoss() # 多分类
    # criterion = torch.nn.MSELoss(reduction="mean") # 单回归RMSE
    criterion = torch.nn.L1Loss() # 回归MAE
    with torch.no_grad():
        valid_loss = 0
        count = 0
        for j, (input, label, _, _) in enumerate(valid_loader):
            input_transformer = input.to(device)
            output_tranformer = transformer(input_transformer)
            output_mlp = mlp(output_tranformer)

            # label = label.reshape(-1, 1) # 单分类
            output_mlp = output_mlp.to(torch.float32)
            label = label.to(torch.float32).to(device)
            # loss = F.binary_cross_entropy(output_mlp, label) # 单分类
            # loss = CELoss(output_mlp, label) # 多分类
            # loss = torch.sqrt(criterion(output_mlp, label)) # 单回归RMSE
            loss = criterion(output_mlp, label) # 回归MAE
            valid_loss += loss.item()
            count += 1
        valid_loss /= count
    mlp.train()
    transformer.train()
    return valid_loss


def train_mlp(mlp, transformer, train_loader, valid_loader, test_loader, config, device, predict_dataset):
    transformer.train()
    mlp.train()
    #  
    optimizer = optim.Adam([{'params': transformer.parameters(), 'lr': float(config['lr_finetune'])}, {'params': mlp.parameters(), 'lr': float(config['lr'])}])
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
    best_valid_loss = 10000
    # CELoss = torch.nn.CrossEntropyLoss() # 多分类
    # criterion = torch.nn.MSELoss(reduction="mean") # 单回归RMSE
    criterion = torch.nn.L1Loss() # 回归MAE
    for i in range(config['epochs']):
        loss_sum = 0
        for j, (input, label, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            input_transformer = input.to(device)
            output_tranformer = transformer(input_transformer)
            output_mlp = mlp(output_tranformer)
            label = label.reshape(-1, 1) # 单分类
            output_mlp = output_mlp.to(torch.float32)
            label = label.to(torch.float32).to(device)
            # loss = F.binary_cross_entropy(output_mlp, label) # 单分类
            # loss = CELoss(output_mlp, label) # 多分类
            # loss = torch.sqrt(criterion(output_mlp, label)) # 回归RMSE
            loss = criterion(output_mlp, label) # 回归MAE
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (j+1) % config['print_every_epochs'] == 0:
                print('Training... epoch: {} , iter: {} , loss: {} '.format(i, j+1, loss_sum/config['print_every_epochs']))
                loss_sum = 0
                loss = evaluate_mlp(mlp, transformer, valid_loader, device)
                if loss < best_valid_loss:
                    print('saving... epoch:{}, iter:{}, best_valid_loss:{} '.format(i, j+1, loss))
                    torch.save(mlp.state_dict(), 'procedure_model/mlp_best_'+config['dataset']+'.pkl')
                    torch.save(transformer.state_dict(), 'procedure_model/transformer_best_'+config['dataset']+'.pkl')
                    best_valid_loss = loss
                    predict_dataset.is_mlp = 0
                    # pred_list, true_list = predict_mlp(mlp, transformer, test_loader, config, device) # 分类
                    predict_loss = predict_mlp(mlp, transformer, test_loader, config, device) # 回归
                    predict_dataset.is_mlp = 2
                    # precision, recall, therashold= precision_recall_curve(true_list, pred_list)
                    # aupr = auc(recall, precision)
                    # print('The AUPR is {} in the procedure'.format(aupr))
                    # print('The auc of roc is {} in the procedure'.format(roc_auc_score(true_list, pred_list))) # 分类
                    print('The rmse is {} in the procedure'.format(predict_loss)) # 回归

        
    print('train mlp complete!')


def predict_mlp(mlp, transformer, test_loader, config, device):
    pred_list =[]
    true_list = []
    mlp.eval()
    transformer.eval()
    # criterion = torch.nn.MSELoss(reduction="mean") # 单回归RMSE
    criterion = torch.nn.L1Loss() # 回归MAE
    with torch.no_grad():
        count = 0
        predict_loss = 0
        for j, (input, label, _, _) in enumerate(test_loader):
            input_transformer = input.to(device)
            output_tranformer = transformer(input_transformer)
            output_mlp = mlp(output_tranformer)

            # output_mlp = output_mlp.cpu().tolist() # 分类
            # label = label.tolist()
            # for i in output_mlp:
            #     pred_list.append(i)
            # for i in label:
            #     true_list.append(i) # 分类 这里需要把在两个分类注释之间的代码全注释掉，而其他的只需注释那一行

            label = label.to(device) # 回归
            # loss = torch.sqrt(criterion(output_mlp, label)) # 回归RMSE
            loss = criterion(output_mlp, label) # 回归MAE
            predict_loss += loss.item()
            count += 1
        predict_loss /= count # 回归
    mlp.train()
    transformer.train()
    # return pred_list, true_list # 分类
    return predict_loss # 单回归


def seed_all(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def predict(task):
    # seed = 8
    # seed_all(seed)

    config = yaml.load(open("config_predict.yaml", "r"), Loader=yaml.FullLoader)
    # predict_dataset = MyDataset(config['dataset'], config['smile_vocab_path'], config['inchi_vocab_path'], config['smile_codes_path'], config['smile_vocab_bpe_path'], config['max_seq_len'])
    predict_dataset = MyDataset(config['dataset'], config['smile_vocab_path'], config['inchi_vocab_path'], config['smile_codes_path'], config['smile_vocab_bpe_path'], config['max_seq_len'], task)

    train_loader, valid_loader, test_loader, finetune_loader, finetune_valid_loader = get_data_loader(predict_dataset, config['valid_ratio'], config['test_ratio'], config['batch_size'], config['num_workers'])
    print(len(train_loader), len(valid_loader), len(test_loader), len(finetune_loader), len(finetune_valid_loader))

    config['print_every_epochs'] = int(len(train_loader)/2) - 5 # 涉及到task或者train时
    # config['print_every_epochs'] = int(len(finetune_loader)/2) - 5 # finetune时
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformer = MyTransformer(config['batch_size'], config['vocab_size'], config['d_model'], config['max_seq_len'], config['project_dimension']).to(device)
    transformer.load_state_dict(torch.load('valid_model/model_epoch9_iter-1.pkl'), strict=True)
    transformer.eval()

    mlp = MLP(config['project_dimension'], config['max_seq_len']).to(device)
    nt_xent_loss = NTXentLoss(device, config['batch_size'], config['temperature'], True)
    
    # predict_dataset.is_mlp = 1
    # finetune_transformer(transformer, finetune_loader, finetune_valid_loader, nt_xent_loss, config, device)
    transformer.load_state_dict(torch.load('procedure_model/transformer_finetune_'+config['dataset']+'.pkl'))
    transformer.eval()
    
    predict_dataset.is_mlp = 2
    train_mlp(mlp, transformer, train_loader, valid_loader, test_loader, config, device, predict_dataset)
    mlp.load_state_dict(torch.load('procedure_model/mlp_best_'+config['dataset']+'.pkl'))
    mlp.eval()
    transformer.load_state_dict(torch.load('procedure_model/transformer_best_'+config['dataset']+'.pkl'))
    transformer.eval()
    predict_dataset.is_mlp = 0
    # pred_list, true_list = predict_mlp(mlp, transformer, test_loader, config, device) # 分类
    predict_loss = predict_mlp(mlp, transformer, test_loader, config, device) # 回归
    # auc = roc_auc_score(true_list, pred_list) # 分类
    # print('The auc of roc is {}'.format(auc)) # 分类
    # precision, recall, therashold= precision_recall_curve(true_list, pred_list)
    # aupr = auc(recall, precision)
    # print('The AUPR is ', aupr)
    print('The rmse is {}'.format(predict_loss)) # 回归
    # return auc # 分类
    return predict_loss # 回归
    # return aupr

predict('') #


