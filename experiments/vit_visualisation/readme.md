# Attention visualisation of ViT

File `models_modified.py` is an update for `models.py` in root directory. 
The ViT model in `models_modified.py` supports returning the attention weights along with the 
prediction. The format is like:
```python
logit, attn_weights = model(data)
```
The method 'Attention Rollout' has been used to generate the attention visualisation as a mask 
and added on the original picture. 
See [Quantifying Attention Flow in Transformers](https://arxiv.org/pdf/2005.00928.pdf) for details.

The file `vit_selfmade.ipynb` is provided as an example.