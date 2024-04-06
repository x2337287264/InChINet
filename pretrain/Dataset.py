from torch.utils.data import Dataset, DataLoader
from .data_process import load_vocab, read_data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import random
from rdkit import Chem
import fastBPE

PAD = 0
UNK = 1
BOS = 2
EOS = 3
MASK = 4

exculde_aug_char = []


def token_mask(smile):
    num_aug = int(np.floor(0.25 * len(smile)))
    idx_list = random.sample(range(0, len(smile)), num_aug)
    for idx in idx_list:
        i = idx
        while smile[i] in exculde_aug_char:
            i = random.randint(0, len(smile)-1)
            while i in idx_list:
                i = random.randint(0, len(smile)-1)
        smile[i] = MASK
    return smile


def reordering(smile):
    num_aug = int(np.floor(0.25 * len(smile)))  
    idx_list = random.sample(range(0, len(smile)), num_aug)
    temp_idx_list = idx_list[:]
    token_temp_list = []
    k = 0 
    for idx in idx_list:
        i = idx
        while smile[i] in exculde_aug_char:
            i = random.randint(0, len(smile)-1)
            while i in idx_list:
                i = random.randint(0, len(smile)-1)
            temp_idx_list[k] = i
        k += 1
        token_temp_list.append(smile[i])
    random.shuffle(temp_idx_list)
    i = 0
    for idx in temp_idx_list:
        smile[idx] = token_temp_list[i]
        i += 1
    return smile


def enumration(content, smi, vocab, bpe, max_seq_len):
    smile = ''
    for ele in smi:
        smile += ele
    smile = smile.replace('@@', '')
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return content
    num, atoms_list = range(mol.GetNumAtoms()), mol.GetNumAtoms()
    try:
        smile = Chem.MolToSmiles(Chem.RenumberAtoms(mol, random.sample(num, atoms_list)), canonical=False, isomericSmiles=True)
    except RuntimeError:
        print('Runtime Error')
        return content
    except Exception:
        print('Unknown Error')
        return content
    temp_list = []
    temp_list.append(smile)
    smile = bpe.apply(temp_list)[0].split()
    if len(smile) >= max_seq_len-1:
        return content
    content = [vocab.get(ele, UNK) for ele in smile]
    return content


def random_aug(content, smile, vocab, bpe, max_seq_len):
    rn = random.randint(0, 3)
    if rn == 0:
        return token_mask(content)
    elif rn == 1:
        return reordering(content)
    elif rn == 2:
        return enumration(content, smile, vocab, bpe, max_seq_len)
    else:
        return content


class MyDataset(Dataset):
    def __init__(self, processed_smile_directory, tokenized_inchi_directory, smile_vocab_path, inchi_vocab_path, smile_vocab_bpe_path, smile_codes_path, max_seq_len):
        super(MyDataset, self).__init__()
        self.smile_vocab = load_vocab(smile_vocab_path)
        self.inchi_vocab = load_vocab(inchi_vocab_path)
        self.smiles_data = read_data(processed_smile_directory)
        self.inchi_data = read_data(tokenized_inchi_directory)
        self.bpe = fastBPE.fastBPE(smile_codes_path, smile_vocab_bpe_path)
        self.max_len = max_seq_len


    def __getitem__(self, idx):
        smile = self.smiles_data[idx]
        inchi = self.inchi_data[idx]
        smile_content = [self.smile_vocab.get(ele, UNK) for ele in smile]
        inchi_content = [self.inchi_vocab.get(ele, UNK) for ele in inchi]
        while len(smile_content) >= self.max_len-1 or len(inchi_content) >=self.max_len-1:
            idx += 1
            idx = idx % len(self.smiles_data)
            smile = self.smiles_data[idx]
            inchi = self.inchi_data[idx]
            smile_content = [self.smile_vocab.get(ele, UNK) for ele in smile]
            inchi_content = [self.inchi_vocab.get(ele, UNK) for ele in inchi]
        smile_aug = random_aug(smile_content, smile, self.smile_vocab, self.bpe, self.max_len)

        smile_aug = [BOS] + smile_aug + [EOS]
        padding = [PAD] * (self.max_len - len(smile_aug))
        smile_aug.extend(padding)

        inchi_aug = [BOS] + inchi_content + [EOS]
        padding = [PAD] * (self.max_len - len(inchi_aug))
        inchi_aug.extend(padding)
        return torch.tensor(smile_aug), torch.tensor(inchi_aug)

    def __len__(self):
        return len(self.smiles_data)
    

def get_data_loader(pretrain_dataset, valid_ratio, batch_size, num_workers):
    num_smiles = len(pretrain_dataset)
    indices = list(range(num_smiles))
    np.random.shuffle(indices)

    valid_size = int(np.floor(valid_ratio * num_smiles))
    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(pretrain_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, drop_last=True)

    valid_loader = DataLoader(pretrain_dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, drop_last=True)

    return train_loader, valid_loader

# smi = 'CC[N+](C)(C)Cc1ccccc1Br'
# mol = Chem.MolFromSmiles(smi)
# num, atoms_list = range(mol.GetNumAtoms()), mol.GetNumAtoms()
# smile = Chem.MolToSmiles(Chem.RenumberAtoms(mol, random.sample(num, atoms_list)), canonical=False, isomericSmiles=True)
# print(smile)
