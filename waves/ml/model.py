from torch import nn
from waves.ml.util import set_seeds


def create_model(config, device, feature_dim, output_dim):
    """ Create a machine learning model """
    # Make the model according to configuration and available models
    print("Making model...\n")
    print(f"Input dimensions: {feature_dim}")
    #print(f"Layers: {config.hidden_layers}")
    print(f"Units per layers: {config.hidden_units}")
    print(f"Output dimensions: {output_dim}\n")
    print(f"Model architecture: {config.model_architecture}")
    if config.model_architecture == 'NeuralNetwork_Linear_ReLU':
        model = NeuralNetwork_Linear_ReLU(feature_dim, config.hidden_layers,
                                      config.hidden_units, output_dim,
                                      config.random_seed) 
    elif config.model_architecture == 'NeuralNetwork_Conv1D_C1':
        model = NeuralNetwork_Conv1D_C1(config.random_seed)  
    elif config.model_architecture == 'AutoEncoder_Conv1D_AE1':
        model = AutoEncoder_Conv1D_AE1(config.random_seed) 
    elif config.model_architecture == 'FCN':
        model = FCN()
    model.to(device)      # ensure model is on correct device
    print(model) 
    
    return model


class NeuralNetwork_Linear_ReLU(nn.Module):
    """ Linear Layer + ReLU Activation Neural Network Model """
    def __init__(self, feature_dim, hidden_layers, hidden_units, 
                 output_dim, random_seed):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_dim,hidden_units))
        self.layers.append(nn.ReLU())
        for ii in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_units,hidden_units))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_units,output_dim))
        self.layers.append(nn.Softmax(dim=1))
        self.init_weights(random_seed)

        return
    
    def init_weights(self, random_seed):
        print('Initial Layer Weights ---------------------------------')
        for idx,m in enumerate(self.layers):
          if type(m) == nn.Linear:
            set_seeds(random_seed+idx)
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.1)
            print(m)
            print(m.weight)
            print("")
            
        return 
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x   
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GAP1d(nn.Module):
    def __init__(self):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.gap(x)
        x = self.flatten(x)
        return x


class FCN(nn.Module):
    # https://arxiv.org/pdf/1611.06455.pdf
    # Time Series Classification from Scratch with Deep Neural Networks
    # A Strong Baseline
    # Zhiguang Wang, Weizhong Yan, GE GLOBAL RESEARCH
    # Tim Oates, University of Maryland Baltimore
    def __init__(self):
        super(FCN, self).__init__()
        self.convblock1 = ConvBlock(1, 128, kernel_size=7, stride=1, padding=3)
        self.convblock2 = ConvBlock(128, 256, kernel_size=5, stride=1, padding=2)
        self.convblock3 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1)
        self.gap = GAP1d()
        self.fc = nn.Linear(128, 3)
        self.soft=nn.Softmax(dim=1)

    def forward(self, x):
        x=x[:,None,:]
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.fc(x)
        x = self.soft(x)
        
        return x

    
class AutoEncoder_Conv1D_AE1(nn.Module):
    def __init__(self, random_seed):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,10,5),  # Update the number of input channels to match the input tensor
            nn.ReLU(),
            nn.Conv1d(10,5,5),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(5,10,5),
            nn.ReLU(),
            nn.ConvTranspose1d(10,1,5),  # Update the number of input channels to match the input tensor
            )
        self.init_weights(random_seed)
        
    def init_weights(self, random_seed):
        print('Initial Layer Weights ---------------------------------')
        set_seeds(random_seed)
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear)\
            or isinstance(module, nn.ConvTranspose1d):
                nn.init.kaiming_uniform_(module.weight)
                module.bias.data.fill_(0.1)
                print(module)
                print(module.weight)
        
    def forward(self, x):
        x=x[:,None,:]
        encoded = self.encoder(x)
        #encoded=encoded[:,None,:]
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)
    


    
    
# class NeuralNetwork_Conv1D_C1(nn.Module):
#     def __init__(self, random_seed):
#         super().__init__()
#         self.c1=nn.Conv1d(1,100,5)
#         self.c2=nn.Conv1d(100,50,5)
#         self.globavgpool=nn.AvgPool1d(2040) # iota
#         #self.globavgpool=nn.AvgPool1d(237) # theta
#         self.flat=nn.Flatten()
#         self.linear=nn.Linear(50,3)
#         self.soft=nn.Softmax(dim=1)
#         #self.init_weights(random_seed)
        
#     def init_weights(self, random_seed):
#         print('Initial Layer Weights ---------------------------------')
        
#         set_seeds(random_seed)
#         nn.init.kaiming_uniform_(self.c1.weight)
#         self.c1.bias.data.fill_(0.1)
#         print(self.c1)
#         print(self.c1.weight)
        
#         set_seeds(random_seed)
#         nn.init.kaiming_uniform_(self.c2.weight)
#         self.c2.bias.data.fill_(0.1)
#         print(self.c2)
#         print(self.c2.weight)
        
#         set_seeds(random_seed)
#         nn.init.kaiming_uniform_(self.linear.weight)
#         self.c1.bias.data.fill_(0.1)#linear?
#         print(self.linear)
#         print(self.linear.weight)
        
#         return 
    
#     def forward(self, x):
#         x=x[:,None,:]
#         x=nn.functional.relu(self.c1(x))
#         x=nn.functional.relu(self.c2(x))
#         #x=self.c2(x)
#         x=self.globavgpool(x)
#         x=self.flat(x)
#         x=self.linear(x)
#         x=self.soft(x)
      
#         return x
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, d=1):
        super().__init__()
        p = (k // 2) * d  # "same-ish" padding for odd kernel
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    
class NeuralNetwork_Conv1D_C1(nn.Module):
    def __init__(self, random_seed):
        super().__init__()
        self.random_seed = random_seed
        set_seeds(random_seed)

        # --- Feature extractor ---
        self.c1 = ConvBlock(1,   32, k=7, s=1)
        self.c2 = ConvBlock(32,  64, k=7, s=2)
        self.c3 = ConvBlock(64, 128, k=7, s=2)
        self.c4 = ConvBlock(128, 128, k=7, s=1, d=2)

        self.dropout = nn.Dropout(0.2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(128, 3)
        self.soft=nn.Softmax(dim=1) #
        
        # --- Custom weight initialization using the seed ---
        self.init_weights()

    def init_weights(self):
        print("Initializing Layer Weights (seed =", self.random_seed, ")")
        set_seeds(self.random_seed)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x[:, None, :]  # ensure shape (B, 1, L)

        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.dropout(x)
        x = self.gap(x).squeeze(-1)  # (B, C)
        logits = self.linear(x)
        logits = self.soft(logits)
        return logits  # no softmax, return raw logits