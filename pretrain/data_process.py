from rdkit import Chem
import os
import json

# def smile2txt(smile_list, smile_directory, file_count):
#     if os.path.exists(smile_directory) is False:
#         os.mkdir(smile_directory)
#     file_name = 'smile_'+str(file_count*500000+1)+'_'+str((file_count+1)*500000)+'.txt'
#     file_name = smile_directory+'/'+file_name
#     ent = '\n'
#     f = open(file_name, "w")
#     f.write(ent.join(smile_list))
#     f.close()


# def sdf2smile(sdf_directory, smile_directory):
#     smile_list = []
#     count = 0
#     file_count = 0
#     for file in os.listdir(sdf_directory):
#         file = sdf_directory+'/'+file
#         mol_list = Chem.SDMolSupplier(file)
#         for idx, mol in enumerate(mol_list):
#             if mol is None:
#                 continue
#             if idx % 10000 == 0:
#                 print(idx)
#             smile_list.append(Chem.MolToSmiles(mol))
#             count += 1
#             if count == 500000:
#                 smile2txt(smile_list, smile_directory, file_count)
#                 smile_list.clear()  # 清除list
#                 file_count += 1
#                 count = 0
#                 if file_count == 20:
#                     break
#         if file_count == 20:
#             break
#     print(count)


# def build_vacab(smile_directory):
#     smile_list = []
#     tokenized_smile = []
#     vocab = {}
#     ele_idx = 5
#     if os.path.exists('tokenized_smile') is False:
#         os.mkdir('tokenized_smile')
#     for file in os.listdir(smile_directory):
#         smile_file = smile_directory+'/'+file
#         f = open(smile_file, "r")
#         for line in f.readlines():
#             cur_line = line.strip()
#             smile_list.append(cur_line[:])
#         print(len(smile_list))
#         f.close()
#         for smile in smile_list:
#             toks = atomwise_tokenizer(smile)
#             for ele in toks:
#                 if ele not in vocab:
#                     vocab[ele] = ele_idx
#                     ele_idx += 1
#             toks = ' '.join(toks)
#             tokenized_smile.append(toks)
#         ent = '\n'
#         f = open('tokenized_smile/'+file, 'w')
#         f.write(ent.join(tokenized_smile))
#         f.close()
#         tokenized_smile.clear()
#         smile_list.clear()
#     if os.path.exists('vocabulary') is False:
#         os.mkdir('vocabulary')
#     with open('vocabulary/vocab.json', 'w') as f:
#         json.dump(vocab, f)


def load_vocab(vocab_path):
    with open(vocab_path) as f_obj:
        vocab = json.load(f_obj)
    return vocab


# def cal_max_len_of_smile():
#     max_len = 0
#     count = 0
#     smile_list = []
#     for file in os.listdir('tokenized_smile'):
#         smile_file = 'tokenized_smile'+'/'+file
#         f = open(smile_file, "r")
#         for line in f.readlines():
#             cur_line = line.strip()[:].split()
#             if len(cur_line) > max_len:
#                 max_len = len(cur_line)
#                 print(max_len)

def read_data(data_directory):
    data_list = []
    for file in os.listdir(data_directory):
        data_file = data_directory + '/' + file
        f = open(data_file, "r")
        for line in f.readlines():
            cur_line = line.strip()
            data_list.append(cur_line[:].split())
        f.close()
    return data_list


# def del_prefix(inchi_directory):
#     inchi_list = []
#     if os.path.exists('inchis') is False:
#         os.mkdir('inchis')
#     for file in os.listdir(inchi_directory):
#         inchi_file = inchi_directory + '/' + file
#         f = open(inchi_file, 'r')
#         for line in f.readlines():
#             cur_line = line.strip()
#             inchi_list.append(cur_line[6:])
#         print(len(inchi_list))
#         f.close()
#         ent = '\n'
#         f = open('inchis/' + file, 'w')
#         f.write(ent.join(inchi_list))
#         f.close()
#         inchi_list.clear()


# def read_file(file):
#     content_list = []
#     f = open(file, 'r')
#     for line in f.readlines():
#         cur_line = line.strip()
#         content_list.append(cur_line[:])
#     print(len(content_list))
#     f.close()
#     return content_list


# def write_file(file, content_list):
#     ent = '\n'
#     f = open(file, 'w')
#     content = ent.join(content_list)
#     content = content + '\n'
#     f.write(content)
#     f.close()

# build_vocab_smile('vocabulary/vocab_smile.txt')
# read_smiles('tokenized_smile')
# sdf2smile('sdf', 'smile')
# build_vacab('smile')
# vocab = load_vocab('vocabulary/vocab.json')
# print(vocab)

