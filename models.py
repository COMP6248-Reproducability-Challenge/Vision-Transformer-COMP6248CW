# coding: utf-8
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class ViT(nn.Module):
    """
    Vision Transformer
    """

    def __init__(self, input_size, patch_size, num_classes, num_layers=12, hidden_size=768, mlp_size=3072,
                 num_attention_heads=12, num_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.dim = patch_size[0] * patch_size[1] * num_channels
        self.input_size = input_size
        self.patch_num = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])

        self.num_layers = num_layers

        self.patch_pos_emb = PatchAndPosEmb(self.input_size, self.patch_size, hidden_size, num_channels)
        self.encoder = TransformerEncoder(self.dim, heads=num_attention_heads, num_layers=num_layers, mlp_size=mlp_size)
        self.head = nn.Sequential(OrderedDict([
            ('layer_norm', nn.LayerNorm(self.dim)),
            ('head', nn.Linear(self.dim, out_features=num_classes, bias=True))
        ]))

    def forward(self, tokens):
        tokens = self.patch_pos_emb(tokens)
        tokens = self.encoder(tokens)
        tokens = self.head(tokens)
        return tokens[:, 0, :]


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, dim, heads, num_layers, mlp_size):
        super().__init__()
        self.layers = nn.Sequential()
        for n in range(num_layers):
            self.layers.add_module('transformer_layer_' + str(n), TransformerEncoderLayer(dim, heads, mlp_size))

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    It is a little different with the original Transformer one.
    The order of layers follow the paper in ViT.
    """

    def __init__(self, dim, heads, mlp_size):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_size, dim)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.mha(x, x, x)[0]
        x = self.layer_norm_2(x)
        x = x + self.mlp(x)
        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, outputs_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, outputs_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out


class PatchAndPosEmb(nn.Module):
    """
    Patching and Position Embedding
    1 patch and flatten  2 linear project  3 positional embedding
    """

    def __init__(self, input_size, patch_size, hidden_size, num_channels):
        super().__init__()
        self.patch_size = patch_size
        self.dim = patch_size[0] * patch_size[1] * num_channels
        self.input_size = input_size
        self.patch_num = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.lpfp = LPFP(self.dim, hidden_size=hidden_size)

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
        tokens = torch.zeros(batch_size, 1, self.dim).to(x.device)
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
        return out
