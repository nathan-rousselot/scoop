"""
File: load_ascad_data.py

Automatic loading of the ASCADv1 dataset. From: 
Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking.

Author: Anonymous
Date: 2025-01-13
Version: 1.0
"""

import torch
import h5py
import numpy as np

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def load_ascad_data(device, batch_size=32):
    ascad_db = h5py.File('dataset/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r')
    profiling_traces = ascad_db['Profiling_traces']
    attack_traces = ascad_db['Attack_traces']


    metadata_test = attack_traces['metadata']
    pt_test = metadata_test['plaintext']
    key_test = metadata_test['key']

    X = torch.Tensor(np.array(profiling_traces['traces'])).to(device)
    Y = torch.LongTensor(profiling_traces['labels']).to(device)


    X_attack = torch.Tensor(np.array(attack_traces['traces'])).to(device)
    Y_attack = torch.LongTensor(attack_traces['labels']).to(device)

    train_loader = FastTensorDataLoader(X, Y, batch_size=batch_size, shuffle=True)
    attack_loader = FastTensorDataLoader(X_attack, Y_attack, batch_size=batch_size, shuffle=False)

    return train_loader, attack_loader, pt_test, X_attack, Y_attack, key_test