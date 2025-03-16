"""
File: custom_mlp.py

Customized MLP model for hyperparameter search. From:
Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking.

Author: Nathan Rousselot
Date: 2025-01-13
Version: 1.0
"""



import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, n_classes, signal_length, n_linear=2, linear_size=1000, activation='ReLU', input_bn=False, dense_bn=False):
        super(MLPModel, self).__init__()
        self.linear_blocks = nn.ModuleList()
        if n_classes < 0:
            raise ValueError('Number of classes must be greater than 0')
        if signal_length < 0:
            raise ValueError('Signal length must be greater than 0')
        if n_linear < 0:
            raise ValueError('Number of linear layers must be greater than 0')
        if linear_size < 1:
            raise ValueError('Linear size must be greater than 0')
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'SeLU':
            self.activation = nn.SELU()
        elif activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function, must be one of ReLU, SeLU, ELU, Tanh')
        if input_bn not in [True, False]:
            raise ValueError('Input batch norm must be a boolean')
        if dense_bn not in [True, False]:
            raise ValueError('Dense batch norm must be a boolean')
        
        self.signal_length = signal_length
        if input_bn:
            self.linear_blocks.append(nn.BatchNorm1d(1))
        self.linear_blocks.append(nn.Flatten())
        self.linear_blocks.append(nn.Linear(signal_length, linear_size))
        if dense_bn:
            self.linear_blocks.append(nn.BatchNorm1d(linear_size))
        self.linear_blocks.append(self.activation)
        for _ in range(n_linear-1):
            self.linear_blocks.append(nn.Linear(linear_size, linear_size))
            if dense_bn:
                self.linear_blocks.append(nn.BatchNorm1d(linear_size))
            self.linear_blocks.append(self.activation)

        self.classifier = nn.Sequential(
            nn.Linear(linear_size, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        for layer in self.linear_blocks:
            x = layer(x)
        x = self.classifier(x)
        return x