# Folder of Experiments

Folder `transfer_to_small_datasets` contains experiments of fine-tuning on various datasets including `ResNet` and `ViT`.

File `pos_emb_vit_b_32.py` and `rgb_emb_vit_b_32.py` show experiments of positional embedding and RGB filter of embedding.


# Reproduced experiments

Here are most of the experiments in the paper. We will reproduce some of them with check marks.

"<font color=lime>√</font>" indicates that the reproduced experiment is included in this repository.

"<font color=red>×</font>" indicates that the reproduced experiment is NOT included in this repository.

* Build a standard ViT. <font color=lime>√</font>

* Pretrained ViT on JFT and finetune on other datasets comparing with other models like ResNet. <font color=red>×</font>

* Pretraining on different size of datasets of ImageNet, ImageNet-21k, and JFT- 300M. <font color=red>×</font>
* Training on random subsets of 9M, 30M, and 90M as well as the full JFT- 300M dataset. <font color=red>×</font>

* Transfer accuracy with increasing pre-training compute. <font color=red>×</font>

* Research into embedding filters. <font color=lime>√</font>
* Research into positional embedding. <font color=lime>√</font>
* Research into attention distance. <font color=red>×</font>

* The performance of ViT with self-supervision. <font color=red>×</font>
* Transfer pre-trained models to various datasets. <font color=lime>√</font>

* Compare SGD and Adam on ResNet. <font color=red>×</font>

* Test different Transformer shapes. <font color=red>×</font>

* Compare positional embeddings of 1-D, 2-D and relative one. <font color=red>×</font>

* More research on axial attentions. <font color=red>×</font>

* Attention maps. <font color=lime>√</font>

# Information of datasets used in paper

| dataset              | size       | number of images | number of classes | Resolution   |
|----------------------|------------|-----------------:|------------------:|--------------|
| JFT-300M             | Unreleased |      303,000,000 |            18,291 | Unknown      |
| ImageNet 21k         | Unknown    |       14,197,122 |            21,841 | 469*387(avg) |
| ImageNet ILSVRC-2012 | 155GB      |        1,281,167 |             1,000 | 469*387(avg) |
| ImageNet ReaL (val)  | 6GB        |           50,000 |             1,000 | 469*387(avg) |
| CIFAR-10             | 170MB      |           60,000 |                10 | 32*32        |
| CIFAR-100            | 170MB      |           60,000 |               100 | 32*32        |
| Oxford Flowers-102   | 330MB      |            8,189 |               102 | around 500   |
| Oxford IIIT-Pets     | 775MB      |            7,394 |                37 | around 500   |
