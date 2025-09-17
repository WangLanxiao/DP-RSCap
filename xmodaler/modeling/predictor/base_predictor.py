# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY

__all__ = ["BasePredictor"]

@PREDICTOR_REGISTRY.register()
class BasePredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float
    ):
        super(BasePredictor, self).__init__()
        self.logits = nn.Linear(hidden_size, vocab_size)
        # self.logits1 = nn.Linear(hidden_size, hidden_size)
        # self.logits2 = nn.Linear(hidden_size, hidden_size)
        # self.logits3 = nn.Linear(hidden_size, hidden_size)
        # self.logits4 = nn.Linear(hidden_size, 2)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]        
        if self.dropout:  
            hidden_states = self.dropout(hidden_states)
        logits = self.logits(hidden_states)
        return { kfg.G_LOGITS: logits}


        # tree_left=self.logits1(hidden_states)
        # tree_right=self.logits2(hidden_states)
        # tree_center=self.logits3(hidden_states)
        # tree_relation=self.softmax(torch.matmul(tree_left,tree_right.transpose(-1,-2)))
        # tree_result=self.logits4(torch.mul(tree_relation.unsqueeze(-1),tree_center.unsqueeze(-3)))
        # return { kfg.G_LOGITS: logits,kfg.TREE_RESULT: tree_result }