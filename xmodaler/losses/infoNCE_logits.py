# Copyright 2021 JD.com, Inc., JD AI
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class INFONCEWithLogits(nn.Module):
    @configurable
    def __init__(self):
        super(INFONCEWithLogits, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.TREE_RESULT in outputs_dict:
            results = outputs_dict[kfg.TREE_RESULT]
            groundtruth = outputs_dict[kfg.TREE_M].to(torch.float32)
            groundtruth = (groundtruth+groundtruth.transpose(-1, -2))-(groundtruth*groundtruth.transpose(-1, -2))
            temp=0.04
            loss=0.0
            for result in results:
                loss=loss+(-torch.log(
                                    (
                                        torch.exp((groundtruth * result.detach())/ temp).sum()-(groundtruth==0).sum()
                                    ).div(
                                        torch.exp((result.detach() / temp)).sum()
                                    )
                                ))
            ret.update({ 'InfoNCELogits Loss(G)': loss*0.03 })
        return ret