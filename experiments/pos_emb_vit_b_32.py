#!/usr/bin/env python
# coding: utf-8

import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from models import ViTB32

model = ViTB32(pretrained=True)

pos_embed = model.patch_pos_emb.pos_emb.permute(0, 2, 1)
print(pos_embed.shape)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(8, 8))
for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i + 1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((7, 7)).detach().cpu().numpy()
    ax = fig.add_subplot(7, 7, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)
# plt.savefig("pos_emb.png")
plt.show()
