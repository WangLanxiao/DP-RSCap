# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
from xmodaler.config import kfg
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from .build import BACKBONE_REGISTRY
from xmodaler.config import configurable
from PIL import Image
from operator import itemgetter
import open_clip
import torch.nn.functional as F
from ..layers.bert import BertLayer, BertGenerationLayer



__all__ = ["CLIP"]
@BACKBONE_REGISTRY.register()
class CLIP(nn.Module):
    @configurable
    def __init__(self,
                 * ,
                 cls_num : int,
                 hidden_size : int,
                 cls_name : str,
                 img_size=384,
                 ent_layers,
                 cls_layers,
                 featlayer_number,
                 clip_scene,
                 clip_entity,
                 **kwargs
                 ):
        super().__init__()
        # self.backbone = CLIPModel.from_pretrained("pretrain_models/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("pretrain_models/clip-vit-base-patch32")

        model_name = 'ViT-B-32'  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
        self.backbone, _, _= open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        ckpt = torch.load('/data1/wlx/project/open_source_dataset/RemoteCLIP/RemoteCLIP-ViT-B-32.pt', map_location="cpu")
        self.backbone.load_state_dict(ckpt)

        for idx, (_name, _weight) in enumerate(self.backbone.named_parameters()):
            _weight.requires_grad = False

        self.max_pool = nn.MaxPool1d(kernel_size=cls_num, return_indices=True)
        self.class_embeddings = nn.Embedding(cls_num, hidden_size)
        self.soft_embeddings = nn.Embedding(1, hidden_size)
        self.class_names=[]
        with open(cls_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            self.class_names.append(line[:-1].lower().split('_')[0])
        self.img_c=768
        self.ent_layers = ent_layers
        self.cls_layers = cls_layers
        self.featlayer_number = featlayer_number
        self.clip_scene = clip_scene
        self.clip_entity = clip_entity

    @classmethod
    def from_config(cls, cfg):
        ent_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(3)]
        )
        cls_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(3)]
        )
        return {
            "ent_layers": ent_layers,
            "cls_layers": cls_layers,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "cls_num": cfg.MODEL.CLASS_NUM,
            "cls_name": cfg.MODEL.CLASS_NAME,
            "featlayer_number": cfg.MODEL.F_NUM,
            "clip_scene": cfg.MODEL.CLIP_S,
            "clip_entity": cfg.MODEL.CLIP_E,
        }
    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.BACKBONE_DIM = 768

    def load_weights(self, pretrained_model):
        pass

    def forward(self, batched_inputs, mode='img'):
        # modified /data1/wlx/anaconda3/envs/xmodar/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py
        if mode=='img':
            x=batched_inputs[kfg.IMG_INPUT]
            # sub=batched_inputs['IMG_PATH']
            # x = torch.stack([self.preprocess(Image.open(i)).cuda() for i in sub],dim=0)
            clip_img,_, img_features_r = self.backbone.encode_image(x)
            if self.featlayer_number==3:
                b1 = img_features_r  #[:,1:,:]
                b2 = torch.nn.functional.adaptive_avg_pool2d(img_features_r.reshape(-1,7,7,self.img_c).permute(0,3,1,2),(5,5)) \
                    .reshape(-1,self.img_c,25).permute(0,2,1)
                b3 = torch.nn.functional.adaptive_avg_pool2d(img_features_r.reshape(-1,7,7,self.img_c).permute(0,3,1,2),(3,3)) \
                    .reshape(-1,self.img_c,9).permute(0,2,1)
            elif self.featlayer_number==2:
                b1 = img_features_r  #[:,1:,:]
                b2 = torch.nn.functional.adaptive_avg_pool2d(img_features_r.reshape(-1,7,7,self.img_c).permute(0,3,1,2),(5,5)) \
                    .reshape(-1,self.img_c,25).permute(0,2,1)
                b3 = None
            elif self.featlayer_number==1:
                b1 = img_features_r  #[:,1:,:]
                b2 = None
                b3 = None
            return {kfg.ATT_FEATS: b1, kfg.PYR_FEATS: [b1,b2,b3], 'CLIP_IMG': clip_img}
        elif mode=='clip':
            pfeats = batched_inputs[kfg.PYR_FEATS]
            p1 = pfeats[0]
            p2 = pfeats[1]
            p3 = pfeats[2]
            gfeats = pfeats[3]

            cls_pre = batched_inputs['CLS_PRE']
            _, idx = self.max_pool(cls_pre)
            # cls_feat = self.class_embeddings(idx)
            bs = cls_pre.shape[0]
            cls_name = list(itemgetter(*idx.squeeze(-1).tolist())(self.class_names))
            # cls_name=['sparseresidential']

            if self.clip_entity:
                inputs_text1 = self.tokenizer(batched_inputs['ENTITY_NAME']).cuda()
                clip_entity, entity_features_r = self.backbone.encode_text(inputs_text1)

            if self.clip_scene:
                begin_promt = 'A image of '
                end_promt = ' scene.'
                new_y = list(map(lambda x, y : begin_promt + x + end_promt, cls_name, cls_name))
                inputs_text2 = self.tokenizer(new_y).cuda()
                cls_entity, entity_features_c = self.backbone.encode_text(inputs_text2)

            Prompt_decoder = self.soft_embeddings.weight.unsqueeze(0).repeat(bs, 1, 1)  # torch.cat([cls_entity.unsqueeze(1), clip_entity.unsqueeze(1)], dim=1)

            return {'CLIP_CLS': entity_features_c, 'CLIP_ENTITY': entity_features_r, 'CLIP_ENTITY_L': Prompt_decoder,
                    'CLIP_SIZE': Prompt_decoder.shape[1], 'CLS_RESULT': idx, kfg.ATT_FEATS: p3,
                    kfg.PYR_FEATS: [p1, p2, p3, gfeats]}
            #

            # soft_feat = self.soft_embeddings.weight.unsqueeze(0).repeat(bs,1,1)

            # if p3 != None:
            #     p3 = torch.cat([entity_features_c, p3], dim=1)
            # if p2 != None:
            #     p2 = torch.cat([entity_features_c, p2], dim=1)
            # if p1 != None:
            #     p1 = torch.cat([entity_features_c, p1], dim=1)



            # Merge = []
            # if p1 != None:
            #     Merge.append(p1)
            # if p2 != None:
            #     Merge.append(p2)
            # if p3 != None:
            #     Merge.append(p3)
            #
            # PP = torch.cat(Merge, dim=1)
            # ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS][:, :, :, 0:1]
            # maks_1 = ext_vmasks.repeat(1, 1, 1, PP.shape[1])
            #
            # if self.clip_entity:
            #     entyty_mask  = ext_vmasks.repeat(1, 1, 1, entity_features_r.shape[1])
            #     for layer_module in self.ent_layers:
            #         entity_features_r = layer_module(entity_features_r, PP, entyty_mask, maks_1)
            #     Prompt_L = entity_features_r
            # else:
            #     Prompt_L = None
            #
            # if self.clip_scene:
            #     cls_mask = ext_vmasks.repeat(1, 1, 1, entity_features_c.shape[1])
            #     for layer_module in self.cls_layers:
            #         entity_features_c = layer_module(entity_features_c, PP, cls_mask, maks_1)
            #     Prompt_C = entity_features_c
            # else:
            #     Prompt_C = None
            #
            # Prompt_decoder = self.soft_embeddings.weight.unsqueeze(0).repeat(bs,1,1)  #torch.cat([cls_entity.unsqueeze(1), clip_entity.unsqueeze(1)], dim=1)
            #
            # return {'CLIP_CLS': Prompt_C, 'CLIP_ENTITY': Prompt_L, 'CLIP_ENTITY_L': Prompt_decoder, 'CLIP_SIZE': Prompt_decoder.shape[1], 'CLS_RESULT': idx, kfg.ATT_FEATS: p3, kfg.PYR_FEATS: [p1,p2,p3,gfeats]}




