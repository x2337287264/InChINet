from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from pretrain.data_process import load_vocab, read_data
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import pandas as pd
from scaffold_split import scaffold_split
import torch
from pretrain.Dataset import random_aug, token_mask, reordering, enumration
import fastBPE
import math
import deepchem as dc

PAD = 0
UNK = 1
BOS = 2
EOS = 3
MASK = 4


def read_file(file):
    content_list = []
    f = open(file, 'r')
    for line in f.readlines():
        cur_line = line.strip()
        content_list.append(cur_line[:])
    print(len(content_list))
    f.close()
    return content_list


def read_csv(dataset, task):
    if dataset == 'BBBP':
        csv_data = pd.read_csv('predict_dataset/BBBP.csv')
        list_data = csv_data[['smiles', 'p_np']].values.tolist()
        smile_data = read_file('predict_dataset/BBBP_smile.txt')
        inchi_data = read_file('predict_dataset/BBBP_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'HIV':
        csv_data = pd.read_csv('predict_dataset/HIV.csv')
        list_data = csv_data[['smiles', 'HIV_active']].values.tolist()
        smile_data = read_file('predict_dataset/HIV_smile.txt')
        inchi_data = read_file('predict_dataset/HIV_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'Tox21':
        csv_data = pd.read_csv('predict_dataset/Tox21.csv')
        # list_data = csv_data[['smiles']].values.tolist()
        list_data1 = csv_data[['smiles', task]].values.tolist()
        list_data = []
        for content in list_data1:
            if np.isnan(content[1]):
                continue
            list_data.append(content)
        smile_data = read_file('predict_dataset/Tox21_smile.txt')
        inchi_data = read_file('predict_dataset/Tox21_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'ClinTox':
        csv_data = pd.read_csv('predict_dataset/clintox.csv')
        list_data = csv_data.values.tolist()
        smile_data = read_file('predict_dataset/ClinTox_smile.txt')
        inchi_data = read_file('predict_dataset/ClinTox_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'ESOL':
        csv_data = pd.read_csv('predict_dataset/ESOL.csv')
        list_data = csv_data[['smiles', 'measured log solubility in mols per litre']].values.tolist()
        smile_data = read_file('predict_dataset/ESOL_smile.txt')
        inchi_data = read_file('predict_dataset/ESOL_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'FreeSolv':
        csv_data = pd.read_csv('predict_dataset/FreeSolv.csv')
        list_data = csv_data[['smiles', 'expt']].values.tolist()
        smile_data = read_file('predict_dataset/FreeSolv_smile.txt')
        inchi_data = read_file('predict_dataset/FreeSolv_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'Lipo':
        csv_data = pd.read_csv('predict_dataset/Lipo.csv')
        list_data = csv_data[['smiles', 'exp']].values.tolist()
        smile_data = read_file('predict_dataset/Lipo_smile.txt')
        inchi_data = read_file('predict_dataset/Lipo_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'clustering':
        list_data = []
        smile_data = read_file('predict_dataset/clustering_smile.txt')
        inchi_data = read_file('predict_dataset/clustering_inchi.txt')
        list_data = smile_data
        return list_data, smile_data, inchi_data
    elif dataset == 'QM7':
        csv_data = pd.read_csv('predict_dataset/qm7.csv')
        list_data = csv_data[['smiles', 'u0_atom']].values.tolist()
        smile_data = read_file('predict_dataset/QM7_smile.txt')
        inchi_data = read_file('predict_dataset/QM7_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'QM8':  
        csv_data = pd.read_csv('predict_dataset/qm8.csv')
        # list_data = csv_data[['smiles']].values.tolist()
        list_data1 = csv_data[['smiles', task]].values.tolist()
        list_data = []
        for content in list_data1:
            if np.isnan(content[1]):
                continue
            list_data.append(content)
        smile_data = read_file('predict_dataset/QM8_smile.txt')
        inchi_data = read_file('predict_dataset/QM8_inchi.txt')
        return list_data, smile_data, inchi_data
    elif dataset == 'QM9':
        csv_data = pd.read_csv('predict_dataset/qm9.csv')
        # list_data = csv_data[['smiles']].values.tolist()
        list_data1 = csv_data[['smiles', task]].values.tolist()
        list_data = []
        for content in list_data1:
            if np.isnan(content[1]):
                continue
            list_data.append(content)
        smile_data = read_file('predict_dataset/QM9_smile.txt')
        inchi_data = read_file('predict_dataset/QM9_inchi.txt')
        return list_data, smile_data, inchi_data

# def get_train_content(predict_data, smile_data, inchi_data, inchi_vocab, bpe, smile_vocab, max_seq_len): # 涉及inchi
def get_train_content(predict_data, bpe, smile_vocab, max_seq_len):
    train_content = []
    for smi in predict_data:
        smile = smi[0]
        temp_list = []
        temp_list.append(smile)
        toks = bpe.apply(temp_list)[0].split()
        smile_content = [smile_vocab.get(ele, UNK) for ele in toks]

        # for z in range(len(smile_data)):                                       # using inchi
        #     if smile_data[z] == smile:
        #         inchi = inchi_data[z].split()
        #         inchi_content = [inchi_vocab.get(ele, UNK) for ele in inchi]
        #         smile_content = inchi_content
        #         break                                                          # using inchi
        train_content.append(smile_content)
        # train_content.append(token_mask(smile_content))
        # train_content.append(reordering(smile_content))
        # train_content.append(enumration(smile_content, toks, smile_vocab, bpe, max_seq_len))
    print(len(train_content))
    return train_content


class MyDataset(Dataset):
    def __init__(self, dataset, smile_vocab_path, inchi_vocab_path, smile_codes_path, smile_vocab_bpe_path, max_seq_len, task, is_mlp=0):
        self.smile_vocab = load_vocab(smile_vocab_path)
        self.inchi_vocab = load_vocab(inchi_vocab_path)
        self.bpe = fastBPE.fastBPE(smile_codes_path, smile_vocab_bpe_path)
        self.predict_data, self.smile_data, self.inchi_data = read_csv(dataset, task)
        self.max_len = max_seq_len
        self.dataset_name = dataset
        self.is_mlp = is_mlp
        # self.train_content = get_train_content(self.predict_data, self.smile_data, self.inchi_data, self.inchi_vocab, self.bpe, self.smile_vocab, self.max_len) #涉及到inchi
        self.train_content = get_train_content(self.predict_data, self.bpe, self.smile_vocab, self.max_len) 

    def __len__(self):
        return len(self.predict_data)

    def __getitem__(self, idx):
        if self.is_mlp == 0:
            smile = self.predict_data[idx][0]
            temp_list = []
            temp_list.append(smile)
            toks = self.bpe.apply(temp_list)[0].split()
            smile_content = [self.smile_vocab.get(ele, UNK) for ele in toks]

            # for z in range(len(self.smile_data)):                                       # 涉及到inchi
            #     if self.smile_data[z] == smile:
            #         inchi = self.inchi_data[z].split()
            #         inchi_content = [self.inchi_vocab.get(ele, UNK) for ele in inchi]
            #         smile_content = inchi_content
            #         break                                                               # 涉及到inchi
            input = [BOS] + smile_content + [EOS]
            padding = [PAD] * (self.max_len - len(input))
            input.extend(padding)                                                        

            label = self.predict_data[idx][1:]
            smile_aug = [0] * self.max_len
            inchi_aug = [0] * self.max_len

        elif self.is_mlp == 1:
            smile = self.smile_data[idx]
            temp_list = []
            temp_list.append(smile)
            toks = self.bpe.apply(temp_list)[0].split()
            smile_content = [self.smile_vocab.get(ele, UNK) for ele in toks]
            if len(smile_content) > self.max_len-2:
                smile_content = smile_content[:510]

            inchi = self.inchi_data[idx].split()
            inchi_content = [self.inchi_vocab.get(ele, UNK) for ele in inchi]
            if len(inchi_content) > self.max_len-2:
                inchi_content = inchi_content[:510]

            smile_aug = random_aug(smile_content, toks, self.smile_vocab, self.bpe, self.max_len)
            smile_aug = [BOS] + smile_aug + [EOS]
            padding = [PAD] * (self.max_len - len(smile_aug))
            smile_aug.extend(padding)
            
            inchi_aug = [BOS] + inchi_content + [EOS]
            padding = [PAD] * (self.max_len - len(inchi_aug))
            inchi_aug.extend(padding)
            

            input = [0] * self.max_len
            label = [0] * 1 # 根据任务数量而改变

        else:
            train_content = self.train_content[idx]
            if len(train_content) > self.max_len-2:
                train_content = train_content[:510]
            input = [BOS] + train_content + [EOS]
            padding = [PAD] * (self.max_len - len(input))
            input.extend(padding)

            smile_aug = [0] * self.max_len
            inchi_aug = [0] * self.max_len
            label = self.predict_data[math.floor(idx)][1:]
        return torch.tensor(input), torch.tensor(label), torch.tensor(smile_aug), torch.tensor(inchi_aug)


def get_finetune_idx(idx_list, predict_data, smile_data):
    finetune_idx_list = []
    for idx in idx_list:
        if predict_data[idx][0] in smile_data:
            for k, smile in enumerate(smile_data):
                if predict_data[idx][0] == smile:
                    finetune_idx_list.append(k)
    return finetune_idx_list


def get_aug_idx(train_idx):
    aug_idx = []
    for idx in train_idx:
        aug_idx.append(idx)
        # aug_idx.append(idx*3+1)
        # aug_idx.append(idx*3+2)
        # aug_idx.append(idx*4+3)
    return aug_idx


def get_data_loader(predict_dataset, valid_ratio, test_ratio, batch_size, num_workers):
    if predict_dataset.dataset_name == 'BBBP':
        train_idx, valid_idx, test_idx = scaffold_split(predict_dataset, valid_ratio, test_ratio)
        finetune_idx = get_finetune_idx(train_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_valid_idx = get_finetune_idx(valid_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_test_idx = get_finetune_idx(test_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_idx = finetune_idx + finetune_test_idx + finetune_valid_idx
    elif predict_dataset.dataset_name == 'HIV':
        train_idx, valid_idx, test_idx = scaffold_split(predict_dataset, valid_ratio, test_ratio)
        finetune_idx = get_finetune_idx(train_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_valid_idx = get_finetune_idx(valid_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_test_idx = get_finetune_idx(test_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_idx = finetune_idx + finetune_test_idx + finetune_valid_idx
    elif predict_dataset.dataset_name == 'QM7':
        idx = np.array(range(len(predict_dataset))).reshape(-1, 1)
        y_list = []
        for i in range(len(idx)):
            y_list.append(predict_dataset.predict_data[i][1])
        y_list = np.array(y_list).reshape(-1, 1)
        dataset = dc.data.DiskDataset.from_numpy(idx, y_list)
        splitter = dc.splits.SingletaskStratifiedSplitter(0)
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
        train_idx = train_dataset.X.reshape(-1).tolist()
        valid_idx = valid_dataset.X.reshape(-1).tolist()
        test_idx = test_dataset.X.reshape(-1).tolist()
        finetune_valid_idx = get_finetune_idx(valid_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_idx = train_idx + valid_idx + test_idx
    else:
        num_smiles = len(predict_dataset)
        indices = list(range(num_smiles))
        np.random.shuffle(indices)

        valid_size = int(np.floor(valid_ratio * num_smiles))
        test_size = int(np.floor(test_ratio * num_smiles))
        train_idx, valid_idx, test_idx = indices[valid_size+test_size:], indices[:valid_size], indices[valid_size:valid_size+test_size]  # 打乱indices然后获取训练集和验证集的idxs
        finetune_idx = get_finetune_idx(train_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_valid_idx = get_finetune_idx(valid_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_test_idx = get_finetune_idx(test_idx, predict_dataset.predict_data, predict_dataset.smile_data)
        finetune_idx = finetune_idx + finetune_test_idx + finetune_valid_idx

    train_idx = get_aug_idx(train_idx)
    valid_idx = get_aug_idx(valid_idx)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    finetune_sampler = SubsetRandomSampler(finetune_idx)
    finetune_valid_sampler = SubsetRandomSampler(finetune_valid_idx)

    # train_sampler = SequentialSampler(train_idx)  
    # valid_sampler = SequentialSampler(valid_idx)
    # test_sampler = SequentialSampler(test_idx)
    # finetune_sampler = SequentialSampler(finetune_idx)
    # finetune_valid_sampler = SequentialSampler(finetune_valid_idx)
    

    train_loader = DataLoader(predict_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, drop_last=False)
    valid_loader = DataLoader(predict_dataset, batch_size=batch_size, sampler=valid_sampler,
                             num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(predict_dataset, batch_size=batch_size, sampler=test_sampler,
                              num_workers=num_workers, drop_last=False)
    finetune_loader = DataLoader(predict_dataset, batch_size=batch_size, sampler=finetune_sampler,
                             num_workers=num_workers, drop_last=False)
    finetune_valid_loader = DataLoader(predict_dataset, batch_size=batch_size, sampler=finetune_valid_sampler,
                              num_workers=num_workers, drop_last=False)
    
    return train_loader, valid_loader, test_loader, finetune_loader, finetune_valid_loader


