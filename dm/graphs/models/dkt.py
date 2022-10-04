import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..weights_initializer import weights_init


class DKT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")
        self.metric = config.metric
        self.input_dim = config.hidden_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.output_dim = config.output_dim
        self.nonlinearity = config.nonlinearity

        # self.input_dim = config.input_dim
        if self.metric == "rmse":
            self.embed_in = nn.Linear(2, config.hidden_dim)
        else:
            self.embed_in = nn.Embedding(num_embeddings=2 * self.output_dim + 1,
                                          embedding_dim=self.hidden_dim)

        self.rnn_type = config.rnn_type
        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, nonlinearity=self.nonlinearity)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                               batch_first=True)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers,
                              batch_first=True)
        else:
            raise TypeError("RNN type is not supported.")
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        if self.metric == "rmse":
            x = x.float()
        h0 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], device=self.device)
        c0 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], device=self.device)
        # nn.init.xavier_normal_(h0)
        # nn.init.xavier_normal_(c0)

        x = self.embed_in(x)
        if self.rnn_type == "LSTM":
            out, hn = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)
        out = self.fc(out)
        return self.sig(out)
