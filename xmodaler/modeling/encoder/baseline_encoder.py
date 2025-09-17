# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn
from ..layers.scattention import SCAttention
from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer, BertGenerationLayer
from .build import ENCODER_REGISTRY
import torch.nn.functional as F
import numpy as np
import copy
__all__ = ["BaselineEncoder"]

@ENCODER_REGISTRY.register()
class BaselineEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_layers1,
        bert_layers2,
        bert_layers3,
        semcomphder_layers,
        hidden_size: int,
        num_hidden_layers: int,
        num_semcomphder_layers: int,
        slot_size: int,
        cls_num: int,
        num_classes: int,
        max_pos: int,
        feat_py_num: int
    ):
        super(BaselineEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_semcomphder_layers = num_semcomphder_layers
        self.layers1 = bert_layers1
        self.layers2 = bert_layers2
        self.layers3 = bert_layers3
        self.decoder_enc_layers = semcomphder_layers
        self.num_classes = num_classes
        self.slot_size = slot_size
        self.max_pos_len = max_pos
        self.softatt = SoftAttention(hidden_size, hidden_size, hidden_size)
        self.semantics_pred = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes+1)   
        )

        self.gvfeat_embed = nn.Sequential(
            nn.Linear(hidden_size * (num_hidden_layers + 1), hidden_size),
            torch.nn.LayerNorm(hidden_size)
        )
        self.embeddings = nn.Sequential(
            nn.Embedding(num_classes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.slot_embeddings = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.slot = nn.Parameter(torch.FloatTensor(1, slot_size, hidden_size))
        nn.init.xavier_uniform_(self.slot)

        self.class_pre =  nn.Linear(hidden_size*feat_py_num, cls_num)
        self.class_embeddings = nn.Parameter(torch.FloatTensor(cls_num, hidden_size))
        self.class_embeddings2 = nn.Embedding(cls_num, hidden_size)
        self.max_pool = nn.MaxPool1d(kernel_size=cls_num,return_indices=True)
        self.softmax = nn.Softmax(dim=-1)
        self.position = nn.Parameter(torch.FloatTensor(self.max_pos_len, hidden_size))
        nn.init.xavier_uniform_(self.position)

    @classmethod
    def from_config(cls, cfg):
        bert_layers1 = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS//3)]
        )
        bert_layers2 = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS//3)]
        )
        bert_layers3 = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS // 3)]
        )
        semcomphder_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.COSNET.NUM_SEMCOMPHDER_LAYERS)]
        )
        return {
            "bert_layers1": bert_layers1,
            "bert_layers2": bert_layers2,
            "bert_layers3": bert_layers3,
            "semcomphder_layers": semcomphder_layers,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "num_semcomphder_layers": cfg.MODEL.COSNET.NUM_SEMCOMPHDER_LAYERS,
            "slot_size": cfg.MODEL.COSNET.SLOT_SIZE,
            "num_classes": cfg.MODEL.COSNET.NUM_CLASSES,
            "cls_num": cfg.MODEL.CLASS_NUM,
            "max_pos": cfg.MODEL.COSNET.MAX_POS,
            "feat_py_num": cfg.MODEL.F_NUM
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.COSNET = CN()
        cfg.MODEL.COSNET.NUM_SEMCOMPHDER_LAYERS = 3
        cfg.MODEL.COSNET.SLOT_SIZE = 6
        cfg.MODEL.COSNET.NUM_CLASSES = 906
        cfg.MODEL.COSNET.MAX_POS = 26
        cfg.MODEL.COSNET.FILTER_WEIGHT = 1.0
        cfg.MODEL.COSNET.RECONSTRUCT_WEIGHT = 0.1

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            pfeats = batched_inputs[kfg.PYR_FEATS]
            p1=pfeats[0]
            p2=pfeats[1]
            p3=pfeats[2]

            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS][:,:,:,0:1]
            gfeats=[]

            if p1!=None:
                maks_1 = ext_vmasks.repeat(1, 1, 1, p1.shape[1])
                for layer in [self.layers1,self.layers2,self.layers3]:
                    for layer_module in layer:
                        p1, scor_1 = layer_module(p1, maks_1)
                gfeats.append(p1[:, 0])

            if p2 != None:
                maks_2 = ext_vmasks.repeat(1, 1, 1, p2.shape[1])
                for layer in [self.layers1,self.layers2,self.layers3]:
                    for layer_module in layer:
                        p2, scor_2 = layer_module(p2, maks_2)
                gfeats.append(p2[:, 0])

            if p3 != None:
                maks_3 = ext_vmasks.repeat(1, 1, 1, p3.shape[1])
                for layer in [self.layers1, self.layers2, self.layers3]:
                    for layer_module in layer:
                        p3, scor_3 = layer_module(p3, maks_3)
                gfeats.append(p3[:, 0])



            gfeats = torch.concatenate(gfeats, dim=-1)

            cls_pre = self.class_pre(gfeats)
          
            ret.update({ kfg.ATT_FEATS: p3, kfg.PYR_FEATS: [p1,p2,p3,gfeats], kfg.GLOBAL_FEATS:gfeats,'CLS_PRE': cls_pre})
         
        return ret

class SoftAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(SoftAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        # self.attn_net = SCAttention(hidden_size, 0.1)
        self.v1 = nn.Linear(hidden_size, att_size)
        self.v2 = nn.Linear(hidden_size, att_size)
        self.m1 = nn.Linear(feat_size, att_size)
        self.m2 = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1)
        self.wc = nn.Linear(att_size, att_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.tanh = nn.Tanh()
        self.dp=nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats, query):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        # BB, LL, CC = feats.shape
        v1 = self.v1(feats)
        v2 = self.v2(query.mean(-2))
        v = self.dp(v1*v2)
        m1 = self.m1(feats)
        m2 = self.m2(query.mean(-2))
        m =  self.dp(m1*m2)
        beta_r = F.softmax(self.wa(m), dim=-2)
        beta_c = self.sigmoid(self.wc(m))
        att_feats = beta_c*(beta_r*v)
        return att_feats
