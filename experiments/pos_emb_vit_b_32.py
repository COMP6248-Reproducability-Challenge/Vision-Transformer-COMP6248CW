#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from torchvision.models import vit_b_32

model = vit_b_32(pretrained=True)
pos_embed = model.encoder.pos_embedding
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(8, 8))
for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i + 1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((7, 7)).detach().cpu().numpy()
    ax = fig.add_subplot(7, 7, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)
plt.savefig("pos_emb.png")


# In[ ]:




