import torch
import torch.nn as nn


class MyTransformer(nn.Module):
    def __init__(self, batch_size, vocab_size, d_model, max_seq_len, project_dimension):
        super(MyTransformer, self).__init__()
        self.batch_size = batch_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = self.positionalEmbedding(max_seq_len, d_model)
        # self.bn = nn.BatchNorm1d(max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=12, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.project_head = nn.Linear(d_model, project_dimension)
        # self.dropout = nn.Dropout(p=0.2)

    def positionalEmbedding(self, max_seq_len, d_model):
        pos_matrix = torch.arange(max_seq_len).reshape((-1, 1))  # mx1
        i_matrix = torch.pow(10000, torch.arange(0, d_model, 2).reshape((1, -1)) / d_model)  # 1xn
        pe_table = torch.zeros(max_seq_len, d_model)
        pe_table[:, 0::2] = torch.sin(pos_matrix / i_matrix)
        pe_table[:, 1::2] = torch.cos(pos_matrix / i_matrix)
        pos_embed = nn.Embedding(max_seq_len, d_model)
        pos_embed.weight = nn.Parameter(pe_table, requires_grad=False)
        return pos_embed

    def forward(self, x):
        # (B,T)
        smile_embed = self.embed(x)  # (B,T,H)
        smile_pos = torch.cat([torch.arange(x.shape[1]).reshape((1, -1)) for _ in torch.arange(x.shape[0])])
        cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        smile_pos = smile_pos.to(cuda)
        smile_pe = self.pos_embed(smile_pos)
        encoder_input = smile_embed+smile_pe
        output = self.encoder(encoder_input)
        output = self.project_head(output)
        return output

