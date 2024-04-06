import sys
sys.path.append('..')
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, max_seq_len):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim/2))
        self.bn1 = nn.BatchNorm1d(max_seq_len)
        self.fc2 = nn.Linear(int(input_dim/2), 1)
        self.fc3 = nn.Linear(max_seq_len, int(max_seq_len/2))
        self.bn2 = nn.BatchNorm1d(int(max_seq_len/2))
        self.fc4 = nn.Linear(int(max_seq_len/2), 1)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        self.max_seq_len = max_seq_len
    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.bn1(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.bn1(x)
        # x = self.dropout(x)

        x = self.fc3(x.reshape(-1, self.max_seq_len))
        # x = self.relu(x)
        # x = self.bn2(x)
        # x = self.dropout(x)

        x = self.fc4(x)
        # x = self.sigmoid(x) # 单分类，多分类和回归只有fc4即可
        # x = self.tanh(x)
        return x
