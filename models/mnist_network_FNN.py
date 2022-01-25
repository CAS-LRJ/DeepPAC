import torch.nn as nn

class Network(nn.Module):
    def __init__(self, net_dims, activation=nn.ReLU):
        super(Network, self).__init__()

        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))

            # use activation function if not at end of layer
            if i != len(net_dims) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x=x.reshape(-1,28*28)
        return self.net(x)

def FNN1():
    return Network([784,50,50,50,10])

def FNN2():
    return Network([784]+[100]*3+[10])

def FNN3():
    return Network([784]+[200]*3+[10])

def FNN4():
    return Network([784]+[200]*6+[10])

def FNN5():
    return Network([784]+[200]*9+[10])

def FNN6():
    return Network([784]+[500]*6+[10])