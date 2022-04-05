# Introduction
This repository is for the group course work of COMP6248.

# Plan
The aim is to reproduce the code of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy).

First meeting: 10-April. The paper should have been read before the meeting.

Notice that the hand-in deadline is 13-May-2022 16:00.

## Experiments

Here are most of the experiments in the paper. We will reproduce some experiments with check marks.
* Build a standard ViT. √ 
* Pretrained ViT on JFT and finetune on other datasets comparing with other models like ResNet.
![1](img/1.png)
* Pretraining on different size of datasets of ImageNet, ImageNet-21k, and JFT-
300M.
* Training on random subsets of 9M, 30M, and 90M as well as the full JFT-
300M dataset.
![2](img/2.png)
* Transfer accuracy with increasing pre-training compute.
![3](img/3.png)
* Research into embedding filters.
* Research into positional embedding.
* Research into attention distance.
![4](img/4.png)
* The performance of ViT with self-supervision.
* Compare SGD and Adam on ResNet.
![](img/5.png)
* Test different Transformer shapes.
![](img/6.png)
* Compare positional embeddings of 1-D, 2-D and relative one.
![](img/7.png)
* More research on axial attentions.
![](img/8.png)
* Attention maps.
![](img/9.png)
