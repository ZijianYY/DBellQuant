import torch
import torch.nn as nn
from functools import partial
from smoothquant.fake_quant import W8A8Linear


DEV = torch.device('cuda:0')

def find_layers(module, layers=[nn.Conv2d, nn.Linear, W8A8Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
