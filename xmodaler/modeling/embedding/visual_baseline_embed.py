# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.utils.initialization import trunc_normal_
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY
import torch.nn.functional as F


__all__ = ["VisualBaselineEmbedding"]

@EMBEDDING_REGISTRY.register()
class VisualBaselineEmbedding(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        in_dim: list,
        g_in_dim: int,
        out_dim: int,
        feat_py_num: int,
        **kwargs
    ):
        super(VisualBaselineEmbedding, self).__init__()
        self.fc1 = nn.Linear(in_dim[0], out_dim)
        self.fc2 = nn.Linear(in_dim[1], out_dim)
        self.fc3 = nn.Linear(in_dim[2], out_dim)
        self.feat_py_num = feat_py_num
        # self.embeddings_layers = nn.ModuleList(
        #     [nn.Linear(in_dim, out_dim) for _ in range(3)]
        # )
        self.g_embeddings = nn.Linear(in_dim[2], out_dim) if g_in_dim > 0 else None
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "g_in_dim": cfg.MODEL.VISUAL_EMBED.G_IN_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "feat_py_num": cfg.MODEL.F_NUM
        }

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        embeddings_pos = nn.Parameter(
            torch.zeros(1, cfg.DATALOADER.MAX_FEAT_NUM, cfg.MODEL.VISUAL_EMBED.OUT_DIM))
        trunc_normal_(embeddings_pos, std=.02)
        kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.PYR_FEATS]
        # feats0=self.fc0(self.conv0(feats[0]))
        # feats1=self.fc1(self.conv1(feats[1]))
        # feats2=self.fc2(self.conv2(feats[2]))
        # feats3=self.fc3(self.conv3(feats[3]))
        # embeddings= torch.stack([feats0,feats1,feats2,feats3], dim=1)


        # if self.feat_py_num == 3:
        #     feats1 = self.fc1(feats[-3])
        #     feats2 = self.fc2(feats[-2])
        #     feats3 = self.fc3(feats[-1])
        # elif self.feat_py_num == 2:
        #     feats1 = None
        #     feats2 = self.fc2(feats[-2])
        #     feats3 = self.fc3(feats[-1])
        # elif self.feat_py_num == 1:
        #     feats1 = None
        #     feats2 = None
        #     feats3 = self.fc3(feats[-1])

        if feats[-3]!=None:
            feats1 = self.fc1(feats[-3])
            bs=feats1.shape[0]
        else:
            feats1 = None

        if feats[-2] != None:
            feats2 = self.fc2(feats[-2])
            bs = feats2.shape[0]
        else:
            feats2 = None

        if feats[-1] != None:
            feats3 = self.fc3(feats[-1])
            bs = feats3.shape[0]
        else:
            feats3 = None

        M=[feats1,feats2,feats3]

        # embeddings= torch.stack([feats2,feats3], dim=1)

        if self.g_embeddings is not None:
            gfeats = batched_inputs[kfg.GLOBAL_FEATS].view(bs, 1, 1, -1)
            g_embeddings = self.g_embeddings(gfeats)

            for idx in range(len(M)):
                if M[idx] != None:
                    M[idx] = torch.cat([g_embeddings.squeeze(1), M[idx] ], dim=1)

            M.append(g_embeddings)
            # embeddings = torch.cat([g_embeddings.repeat(1, feats3.shape[1], 1, 1), embeddings], dim=2)

        # embeddings_pos = self.embeddings_pos
        # embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            for idx in range(len(M)):
                if M[idx] != None:
                    M[idx] = self.embeddings_act(M[idx])
            # embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            for idx in range(len(M)):
                if M[idx] != None:
                    M[idx] = self.embeddings_act(M[idx])
            # embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            for idx in range(len(M)):
                if M[idx] != None:
                    M[idx] = self.embeddings_act(M[idx])
            # embeddings = self.embeddings_dropout(embeddings)

        return { kfg.ATT_FEATS: M[-2], kfg.PYR_FEATS: M[:-1], kfg.GLOBAL_FEATS:M[-1] }