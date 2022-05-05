# coding: utf-8
#
import os

import torch
import torch.nn as nn

from pretrained_processor import *


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
        self.encoder = TransformerEncoder(self.dim,
                                          heads=num_attention_heads,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size,
                                          mlp_size=mlp_size)
        self.head = nn.Linear(hidden_size, out_features=num_classes, bias=True)

    def forward(self, tokens):
        tokens = self.patch_pos_emb(tokens)
        tokens = self.encoder(tokens)
        tokens = self.head(tokens[:, 0, :])
        return tokens


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, dim, heads, num_layers, hidden_size, mlp_size):
        super().__init__()
        self.layers = nn.Sequential()
        for n in range(num_layers):
            self.layers.add_module('transformer_layer_' + str(n), TransformerEncoderLayer(dim,
                                                                                          heads,
                                                                                          hidden_size,
                                                                                          mlp_size))
        self.layers.add_module('layer_norm', nn.LayerNorm(hidden_size, eps=1e-06))

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    It is a little different with the original Transformer one.
    The order of layers follow the paper in ViT.
    """

    def __init__(self, dim, heads, hidden_size, mlp_size):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.mha = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-06)
        self.mlp = MLP(hidden_size, mlp_size, hidden_size)

    def forward(self, x):
        x_ = self.layer_norm_1(x)
        x_ = x + self.mha(x_, x_, x_)[0]
        x = self.layer_norm_2(x_)
        x = x_ + self.mlp(x)
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

        self.conv = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, hidden_size, 1))
        self.pos_emb = nn.Parameter(torch.rand(1, hidden_size, self.patch_num[0] * self.patch_num[1] + 1))

    def forward(self, x):
        b, c, w, h = x.shape  # b:batch_size  c:channel_size  w:width  h:height
        # Patch, flatten and Linear project
        tokens = self.conv(x)
        tokens = tokens.reshape([b, c * self.patch_size[0] * self.patch_size[1], self.patch_num[0] * self.patch_num[1]])
        # Add cls token
        tokens = torch.cat((self.cls_token.repeat(b, 1, 1), tokens), 2)
        # Positional embedding
        tokens += self.pos_emb
        tokens = tokens.permute(0, 2, 1)
        return tokens


class ViTB16(ViT):
    def __init__(self, pretrained=False):
        super().__init__(input_size=(224, 224), patch_size=(16, 16), num_classes=1000)
        FILENAME = "vit_b_16-c867db91.pth"
        if pretrained:
            if not os.path.exists(FILENAME):
                print("Downloading.")
                download_model(FILENAME, 'https://download.pytorch.org/models/vit_b_16-c867db91.pth')
                print("Download successful, now loading.")
            else:
                print("Pretrained model exists, now loading.")
            state_dict = transfer_pretrained_model(FILENAME)
            self.load_state_dict(state_dict)
            print("Pretrained model loaded.")


class ViTB32(ViT):
    def __init__(self, pretrained=False):
        super().__init__(input_size=(224, 224), patch_size=(32, 32), num_classes=1000)
        FILENAME = "vit_b_32-d86f8d99.pth"
        if pretrained:
            if not os.path.exists(FILENAME):
                print("Downloading.")
                download_model(FILENAME, 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth')
                print("Download successful, now loading.")
            else:
                print("Pretrained model exists, now loading.")
            state_dict = transfer_pretrained_model(FILENAME)
            self.load_state_dict(state_dict)
            print("Pretrained model loaded.")


class ViTL16(ViT):
    def __init__(self, pretrained=False):
        super().__init__(input_size=(224, 224), patch_size=(16, 16), num_classes=1000, hidden_size=1024, mlp_size=4096,
                         num_attention_heads=16,  num_layers=24)
        FILENAME = "vit_l_16-852ce7e3.pth"
        if pretrained:
            if not os.path.exists(FILENAME):
                print("Downloading.")
                download_model(FILENAME, 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth')
                print("Download successful, now loading.")
            else:
                print("Pretrained model exists, now loading.")
            state_dict = transfer_pretrained_model(FILENAME)
            self.load_state_dict(state_dict)
            print("Pretrained model loaded.")


class ViTL32(ViT):
    def __init__(self, pretrained=False):
        super().__init__(input_size=(224, 224), patch_size=(32, 32), num_classes=1000, hidden_size=1024, mlp_size=4096,
                         num_attention_heads=16,  num_layers=24)
        FILENAME = "vit_l_32-c7638314.pth"
        if pretrained:
            if not os.path.exists(FILENAME):
                print("Downloading.")
                download_model(FILENAME, 'https://download.pytorch.org/models/vit_l_32-c7638314.pth')
                print("Download successful, now loading.")
            else:
                print("Pretrained model exists, now loading.")
            state_dict = transfer_pretrained_model(FILENAME)
            self.load_state_dict(state_dict)
            print("Pretrained model loaded.")
