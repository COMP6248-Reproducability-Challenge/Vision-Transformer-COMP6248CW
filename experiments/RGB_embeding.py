#!/usr/bin/env python
# coding: utf-8

# In[61]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from torchvision.models import vit_b_32

model = vit_b_32(pretrained=True)


# In[63]:


weight=model.conv_proj.weight.reshape(-1,3072)
weight_t=weight.T


# In[67]:


import cv2
import numpy as np
from sklearn.decomposition import PCA  
#first 28 principal components
pca_sk = PCA(n_components=28)  
newMat = pca_sk.fit_transform(dataMat)
s=newMat.T
s_re=s.reshape(28,3,32,32)


# In[68]:


import torch
x=torch.Tensor(s_re)


# In[69]:


fig, ax = plt.subplots(nrows=4, ncols=7, figsize=(12, 12),
                   subplot_kw={'xticks': [], 'yticks': []})
k=0
for i in range(4):
    for j in range(7):
        ax[i,j].imshow(x[k].permute(1, 2, 0).detach().numpy()*20)
        k=k+1
plt.show()

