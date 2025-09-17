# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY
import torch.nn.functional as F

@LOSSES_REGISTRY.register()
class BCEWithLogits(nn.Module):
    @configurable
    def __init__(self):
        super(BCEWithLogits, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}

        if kfg.TREE_RESULT in outputs_dict:
            logits = outputs_dict[kfg.TREE_RESULT]
            groundtruth = outputs_dict[kfg.TREE_M]
            groundtruth = (groundtruth + groundtruth.transpose(-1, -2)) - (groundtruth * groundtruth.transpose(-1, -2))
            targets = F.one_hot(groundtruth, num_classes=2).to(torch.float32)
            # b,_,_,N = targets.size()
            # loss = self.bce(logits.reshape(b,-1,N), targets.reshape(b,-1,N))
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([19]).cuda(),
                                                      reduction="mean")
            ret.update({'BCEloss': loss * 0.5})
            
        return ret