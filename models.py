# coding: utf-8
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViT(nn.Module):
    def __init__(self, input_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.dim = patch_size[0] * patch_size[1] * 3
        self.input_size = input_size
        self.patch_num = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])

        self.patch_pos_emb = PatchAndPosEmb(self.input_size, self.patch_size)
        # TODO: Unfinished

    def forward(self, x):
        tokens = self.patch_pos_emb(x)
        # TODO: Unfinished
        return


class PatchAndPosEmb(nn.Module):
    """
    Patching and Position Embedding
    1 patch and flatten  2 linear project  3 positional embedding
    """

    def __init__(self, input_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.dim = patch_size[0] * patch_size[1] * 3
        self.input_size = input_size
        self.patch_num = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.lpfp = LPFP(self.dim, hidden_size=768)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb = nn.Parameter(torch.rand(1, self.patch_num[0] * self.patch_num[1] + 1, self.dim))

    def forward(self, input):
        b, c, w, h = input.shape  # b:batch_size  c:channel_size  w:width  h:height
        # Patch and flatten
        tokens = self.patching(input)
        # Linear project
        tokens = self.lpfp(tokens)
        # Add cls token
        tokens = torch.cat((self.cls_token.repeat(b, 1, 1), tokens), 1)
        # Positional embedding
        tokens += self.pos_emb
        return tokens

    def patching(self, x):
        batch_size, _, _, _ = x.shape
        tokens = torch.zeros(batch_size, 1, self.dim)
        for i in range(0, self.patch_num[0]):
            for j in range(0, self.patch_num[1]):
                token = x[:, :,
                        self.patch_size[0] * i:self.patch_size[0] * (i + 1),
                        self.patch_size[1] * j:self.patch_size[1] * (j + 1)]
                tokens = torch.cat((tokens, token.reshape(batch_size, 1, -1)), 1)
        return tokens[:, 1:]


class LPFP(nn.Module):
    """
    Linear Projection of Flattened Patches
    Basically, it is an MLP with the same dim of input_size and output_size.
    """

    def __init__(self, input_size, hidden_size=768):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out
