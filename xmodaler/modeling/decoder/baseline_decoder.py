# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import random
import torch
import torch.nn as nn
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from .decoder import Decoder
from ..layers.pyr_layer import PYRBlock
from .build import DECODER_REGISTRY

__all__ = ["BaselineDecoder"]

@DECODER_REGISTRY.register()
class BaselineDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        layer_drop: float,
        num_generation_layers: int,
        cls_num: int,
        hidden_size: int,
        pyr_generation_layers
    ):
        super(BaselineDecoder, self).__init__()
        self.num_generation_layers = num_generation_layers
        if self.num_generation_layers > 0:
            self.g_layers = pyr_generation_layers
        self.layer_drop = layer_drop

        self.softmax = nn.Softmax(dim=-1)
        
    @classmethod
    def from_config(cls, cfg):
        pyr_generation_layers = nn.ModuleList(
            [PYRBlock(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "pyr_generation_layers": pyr_generation_layers,
            "cls_num": cfg.MODEL.CLASS_NUM,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "layer_drop": cfg.MODEL.BERT.LAYER_DROP,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        clip_size=batched_inputs['CLIP_SIZE']
        pfeats = batched_inputs[kfg.PYR_FEATS]
        p1=pfeats[0]
        p2=pfeats[1]
        p3=pfeats[2]

        Merge = []
        if p1 != None:
            Merge.append(p1)
        if p2 != None:
            Merge.append(p2)
        if p3 != None:
            Merge.append(p3)

        PP=torch.cat(Merge, dim=1)
        CLIP_CLS=batched_inputs['CLIP_CLS']
        CLIP_ENTITY=batched_inputs['CLIP_ENTITY']

        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS][:, :, :, 0:1]
        maks_p = ext_vmasks.repeat(1, 1, 1, PP.shape[1])
        if CLIP_CLS!=None:
            maks_c = ext_vmasks.repeat(1, 1, 1, CLIP_CLS.shape[1])
        else:
            maks_c = None

        if CLIP_ENTITY!=None:
            maks_e = ext_vmasks.repeat(1, 1, 1, CLIP_ENTITY.shape[1])
        else:
            maks_e = None


        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)

        g_tfeats_arr = []
        g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
        ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
        if len(g_tfeats.size()) == 2:
            g_tfeats = g_tfeats.unsqueeze(1)
        
        if kfg.TIME_STEP in batched_inputs:
            if batched_inputs[kfg.TIME_STEP] ==0:
                time_step = batched_inputs[kfg.TIME_STEP] + clip_size
                ext_g_tmasks = ext_g_tmasks[:, :, 0:time_step, 0:time_step]
            else:
                time_step = batched_inputs[kfg.TIME_STEP]+clip_size
                ext_g_tmasks = ext_g_tmasks[:,:, time_step-1:time_step, 0:time_step]
            if kfg.HISTORY_STATES not in batched_inputs:
                shape = list(g_tfeats.size())
                shape[1] = 0
                history_states = [g_tfeats.new(torch.Size(shape))] * self.num_generation_layers
                batched_inputs[kfg.HISTORY_STATES] = history_states
        else:
            history_states = [None] * self.num_generation_layers

        for i, layer_module in enumerate(self.g_layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)
            dropout_probability = random.uniform(0, 1)
            this_layer_drop = self.layer_drop * (i+1)/len(self.g_layers)
            if self.training and (dropout_probability < this_layer_drop):
                g_tfeats_arr.append(g_tfeats)
            else:
                g_tfeats, gailv = layer_module(
                    g_tfeats,
                    PP,
                    CLIP_CLS,
                    CLIP_ENTITY,
                    ext_g_tmasks,
                    maks_p,
                    maks_c,
                    maks_e,
                    history_states[i]
                )

                g_tfeats_arr.append(g_tfeats)

        if kfg.TIME_STEP not in batched_inputs:
            g_hidden_states = g_tfeats_arr[-1][:,clip_size-1:-1,:]
        else:
            g_hidden_states = g_tfeats_arr[-1][:, -1:, :]

        ret.update({ kfg.G_HIDDEN_STATES: g_hidden_states})#,'CLS_PRE': cls_pre })
        return ret