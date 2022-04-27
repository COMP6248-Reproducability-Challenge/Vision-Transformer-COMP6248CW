# coding: utf-8
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViT(nn.Module):
    """
    Vision Transformer
    """
    def __init__(self, input_size, patch_size, num_classes, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.dim = patch_size[0] * patch_size[1] * 3
        self.input_size = input_size
        self.patch_num = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])

        self.num_layers = num_layers

        self.patch_pos_emb = PatchAndPosEmb(self.input_size, self.patch_size)
        self.encoder = TransformerEncoder(self.dim, 12, num_layers)
        self.mlp_head = MLP(self.dim, 768, num_classes)

    def forward(self, tokens):
        tokens = self.patch_pos_emb(tokens)
        tokens = self.encoder(tokens)
        tokens = self.mlp_head(tokens)
        return tokens[:, 0, :]


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, dim, heads, num_layers):
        super().__init__()
        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.add_module('transformer layer', TransformerEncoderLayer(dim, heads))


    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    It is a little different with the original Transformer one.
    The order of layers follow the paper in ViT.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads)
        self.mlp = MLP(dim, 768, dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x + self.mha(x, x, x)[0]
        x = self.layer_norm(x)
        x = x + self.mlp(x)
        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, outputs_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, outputs_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        # if not self.training:
        #     out = F.softmax(out, dim=1)
        return out


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
