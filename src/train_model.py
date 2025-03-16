"""
File: load_ascad_data.py

Custom training loop for the profiling of a DL-SCA model using Scoop as an optimizer. 
Coming from:
Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking.

WARNING: will not work with other optimizers. Comment out lines 58/59 to use with other optimizers.

Author: Nathan Rousselot
Date: 2025-01-13
Version: 1.0
"""


import torch
import torch.nn.functional as F
import math
import time
import numpy as np

def train_model(model, optimizer, n_epochs, train_loader, valid_loader, hessian_update=10, verbose=False, save_best_model=True, path='best_model.pt', device=None, finetuning=False, entropy=8, MLP=False):
    """
    Train a model.

    :param model: model to train.
    :param optimizer: optimizer to use.
    :param n_epochs: number of epochs to train.
    :param train_loader: training data loader.
    :param valid_loader: validation data loader.

    :returns: training and validation losses.
    """
    train_losses = []
    valid_losses = []
    best_val = torch.inf


    start_time = time.time()
    epoch = 0

    n = len(train_loader)

    while True:
        iter = -1
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            if device is not None:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            if not MLP:
                Y_pred = model(X_batch.unsqueeze(1))
            else:
                Y_pred = model(X_batch)
            loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
            loss.backward(create_graph=True)
            if iter % hessian_update == hessian_update - 1:
                optimizer.hutchinson_hessian() # SCOOP SPECIFIC LINE
            optimizer.step()
            train_loss += loss.item()
            iter += 1
            if verbose:
                print('Iteration ', iter, '/', n, '   Train Loss: ', train_loss/(iter+1), end='\r')
        train_losses.append(train_loss / len(train_loader))
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                if device is not None:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                if not MLP:
                    Y_pred = model(X_batch.unsqueeze(1))
                else:
                    Y_pred = model(X_batch)
                loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
                valid_loss += loss.item()
        valid_losses.append(valid_loss / len(valid_loader))
        if verbose:
            print(f'Epoch {epoch + 1}/{n_epochs} | '
                  f'Train loss: {train_losses[-1]:.4f} | '
                  f'Valid loss: {valid_losses[-1]:.4f} | '
                  'Expected time left: {:.2f} s'.format((n_epochs - epoch - 1) * (time.time() - start_time) / (epoch + 1)), end='\r')
        if valid_losses[-1] < best_val and save_best_model:
            best_val = valid_losses[-1]
            torch.save(model, path)
        epoch += 1
        np.save('train_losses.npy', train_losses)
        np.save('valid_losses.npy', valid_losses)
        if epoch >= n_epochs:
            if not finetuning:
                break
            else:
                if valid_losses[-1] > entropy:
                    break
        if finetuning and valid_losses[-1] > entropy+0.1:
            break
        if epoch > 1 and math.isnan(valid_losses[-1]):
            break
    return train_losses, valid_losses, path