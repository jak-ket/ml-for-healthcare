import copy
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# append the filepath to where torch is installed
sys.path.append('/home/millerm/.local/lib/python3.10/site-packages')
# sys.path.append('/home/username/.local/lib/python3.10/site-packages')

import torch
from torch.utils.data import DataLoader, TensorDataset

def create_datasets(train, test, train_target, test_target, valid_pct=0.1, seed=None):
    """Converts NumPy arrays into PyTorch datsets."""

    train, test, train_target, test_target = train, test, train_target, test_target
    assert len(train)==len(train_target)
    idx = np.arange(len(train))
    trn_idx, val_idx = train_test_split(
        idx, test_size=valid_pct, random_state=seed)
    trn_ds = TensorDataset(
        torch.tensor(train[trn_idx]).float(),
        torch.tensor(train_target[trn_idx]).long())

    val_ds = TensorDataset(
        torch.tensor(train[val_idx]).float(),
        torch.tensor(train_target[val_idx]).long())

    tst_ds = TensorDataset(
        torch.tensor(test).float(),
        torch.tensor(test_target).long())
    return trn_ds, val_ds, tst_ds

def create_loaders(data, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl