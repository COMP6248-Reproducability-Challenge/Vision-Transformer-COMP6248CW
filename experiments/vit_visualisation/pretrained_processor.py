from collections import OrderedDict

import requests
import torch


def download_model(filename, url):
    down_res = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(down_res.content)


def transfer_pretrained_model(filename):
    a = torch.load(filename)
    b = OrderedDict()
    for key, value in a.items():
        if key == 'class_token':
            b['patch_pos_emb.cls_token'] = value.reshape([1, 768, 1])
        elif key == 'encoder.pos_embedding':
            b['patch_pos_emb.pos_emb'] = value.permute(0, 2, 1)
        elif 'conv_proj' in key:
            b[key.replace("conv_proj", "patch_pos_emb.conv")] = value
        elif 'encoder.layers.encoder_layer' in key:
            if 'ln_' in key:
                k = key.replace('encoder.layers.encoder_layer_', 'encoder.layers.transformer_layer_').replace('.ln_',
                                                                                                              '.layer_norm_')
                b[k] = value
            elif 'attention' in key:
                k = key.replace('encoder.layers.encoder_layer_', 'encoder.layers.transformer_layer_').replace(
                    '.self_attention', '.mha')
                b[k] = value
            elif 'mlp' in key:
                k = key.replace('encoder.layers.encoder_layer_', 'encoder.layers.transformer_layer_').replace(
                    'mlp.linear_',
                    'mlp.fc')
                b[k] = value
        elif 'encoder.ln' in key:
            b[key.replace("encoder.ln", "encoder.layers.layer_norm")] = value
        elif 'heads.head' in key:
            b[key.replace("heads.head", "head")] = value
    return b