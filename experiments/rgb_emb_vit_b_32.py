#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import ViTB32
from sklearn.decomposition import PCA

model = ViTB32(pretrained=True)

weight = model.patch_pos_emb.conv.weight.reshape(-1, 3072)
weight_t = weight.T

# First 28 principal components
pca_sk = PCA(n_components=28)
newMat = pca_sk.fit_transform(weight_t.detach().numpy())
x = torch.Tensor(newMat.T.reshape(28, 3, 32, 32))

# Draw
fig, ax = plt.subplots(nrows=4, ncols=7, figsize=(12, 7),
                       subplot_kw={'xticks': [], 'yticks': []})
for i in range(4):
    for j in range(7):
        ax[i, j].imshow(np.sum(x[4 * i + j].permute(1, 2, 0).detach().numpy(), axis=2),
                        cmap='gray')
        ax[i, j].imshow(x[4 * i + j].permute(1, 2, 0).detach().numpy() * 20,
                        alpha=0.5)
plt.show()
