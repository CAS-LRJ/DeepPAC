import torch.nn as nn
import torch
import numpy as np

class ConvSmallNetwork(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(ConvSmallNetwork, self).__init__()

        self.activation=activation()
        self.conv1=nn.Conv2d(1,16,4,2)
        self.conv2=nn.Conv2d(16,32,4,2)
        self.linear1=nn.Linear(5*5*32,100)
        self.linear2=nn.Linear(100,10)
        self.layers=[]
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        self.layers.append(self.linear1)
        self.layers.append(self.linear2)
    
    def load_pyt_file(self, file):
        with open(file, 'r') as f:
            for layer in self.layers:
                str_in=f.readline()
                while(str_in[0]!='['):
                    str_in=f.readline()
                a_i=eval(str_in)
                str_in=f.readline()
                b_i=eval(str_in)
                if isinstance(layer, nn.Conv2d):
                    a_i=np.array(a_i).transpose((3,2,0,1))
                else:
                    a_i=np.array(a_i)
                b_i=np.array(b_i)
                layer.weight=nn.Parameter(torch.tensor(a_i, dtype=torch.float))
                layer.bias=nn.Parameter(torch.tensor(b_i, dtype=torch.float))

    def forward(self, x):
        x=self.activation(self.layers[0](x))
        x=self.activation(self.layers[1](x))
        x=x.reshape(-1,5*5*32)
        x=self.activation(self.layers[2](x))
        x=self.layers[3](x)
        return x

class ConvMedNetwork(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(ConvMedNetwork, self).__init__()

        self.activation=activation()
        self.conv1=nn.Conv2d(1,16,4,2,1)
        self.conv2=nn.Conv2d(16,32,4,2,1)
        self.linear1=nn.Linear(7*7*32,1000)
        self.linear2=nn.Linear(1000,10)
        self.layers=[]
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        self.layers.append(self.linear1)
        self.layers.append(self.linear2)
    
    def load_pyt_file(self, file):
        with open(file, 'r') as f:
            for layer in self.layers:
                str_in=f.readline()
                while(str_in[0]!='['):
                    str_in=f.readline()
                a_i=eval(str_in)
                str_in=f.readline()
                b_i=eval(str_in)
                if isinstance(layer, nn.Conv2d):
                    a_i=np.array(a_i).transpose((3,2,0,1))
                else:
                    a_i=np.array(a_i)
                b_i=np.array(b_i)
                layer.weight=nn.Parameter(torch.tensor(a_i, dtype=torch.float))
                layer.bias=nn.Parameter(torch.tensor(b_i, dtype=torch.float))

    def forward(self, x):
        x=self.activation(self.layers[0](x))
        x=self.activation(self.layers[1](x))
        x=x.reshape(-1,7*7*32)
        x=self.activation(self.layers[2](x))
        x=self.layers[3](x)
        return x