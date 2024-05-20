import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def create_loaders_umap(data, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl

def extract_embeddings(data_loader, model, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            embeddings.append(outputs.detach().cpu().numpy())  
    return np.concatenate(embeddings, axis=0)

def draw_umap(embeddings, colors, name_ds, train, s = 2, alpha = 0.5, random_state = 42):
    n_neighbors_values = [5, 15, 35]
    min_dist_values = [0.1, 0.5, 0.9]
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(name_ds)

    for i, n_neighbors in enumerate(n_neighbors_values):
        for j, min_dist in enumerate(min_dist_values):
            emb = UMAP(n_neighbors = n_neighbors, min_dist = min_dist, random_state = random_state).fit_transform(embeddings)
            
            axs[i, j].scatter(emb[:, 0], emb[:, 1], c = colors, s = s, alpha = alpha)
            axs[i, j].set_title('{}: n_neighbors={}, min_dist={}'.format(name_ds, n_neighbors, min_dist))
            axs[i, j].set_xlabel('Dimension 1')
            axs[i, j].set_ylabel('Dimension 2')
            print('n_neighbors={}, min_dist={}'.format(n_neighbors, min_dist))

    if train == True:
        train = "train"
    else:
        train = "test"

    plt.tight_layout()
    plt.savefig(f"plots/umap_{name_ds}_{train}.png")
    plt.close()