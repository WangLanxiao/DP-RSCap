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
from xmodaler.functional import load_vocab, decode_sequence, decode_sequence_bert
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel


@LOSSES_REGISTRY.register()
class CrossEntropy(nn.Module):
    @configurable
    def __init__(self,vocab_path):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        # self.vocab = load_vocab(vocab_path)
        # self.cli = CLIPModel.from_pretrained("pretrain_models/clip-vit-base-patch32").cuda()
        # self.proce = CLIPProcessor.from_pretrained("pretrain_models/clip-vit-base-patch32")
        # self.cos = nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
        # self.max_pool = nn.MaxPool1d(kernel_size=len(self.vocab), return_indices=True)

        # for idx, (_name, _weight) in enumerate(self.cli.named_parameters()):
        #     _weight.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        return {"vocab_path": cfg.INFERENCE.VOCAB}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.G_TARGET_IDS]

            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1).long()
            loss = self.criterion(logits, targets)
            ret.update({ 'CrossEntropy Loss': loss })

        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            targets = outputs_dict[kfg.U_TARGET_IDS]

            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1).long()
            loss = self.criterion(logits, targets)
            ret.update({'CrossEntropy Loss(U)': loss})

        if 'CLS_PRE' in outputs_dict:
            logits_cls = outputs_dict['CLS_PRE']
            targets_cls = outputs_dict['CLASS_ID']
            loss_cls = self.criterion(logits_cls, targets_cls)
            ret.update({'Class Loss': loss_cls*0.5})

        # _, idx = self.max_pool(outputs_dict[kfg.G_LOGITS])
        # cap = decode_sequence(self.vocab, idx.squeeze(-1))
        # cap_gt = decode_sequence(self.vocab, outputs_dict[kfg.G_TARGET_IDS])
        # img_clip_feat=outputs_dict['CLIP_IMG']
        #
        # inputs_text3 = self.proce(text=cap, return_tensors="pt", padding=True)
        # inputs_text3.data['input_ids'] = inputs_text3.data['input_ids'].cuda()
        # inputs_text3.data['attention_mask'] = inputs_text3.data['attention_mask'].cuda()
        # cap_clip_feat, _ = self.cli.get_text_features(**inputs_text3)
        #
        # inputs_text4 = self.proce(text=cap_gt, return_tensors="pt", padding=True)
        # inputs_text4.data['input_ids'] = inputs_text4.data['input_ids'].cuda()
        # inputs_text4.data['attention_mask'] = inputs_text4.data['attention_mask'].cuda()
        # capgt_clip_feat, _ = self.cli.get_text_features(**inputs_text4)
        #
        # result_loss1 = self.cos(capgt_clip_feat,cap_clip_feat,torch.ones(cap_clip_feat.shape[0]).cuda())
        # result_loss2 = self.cos(img_clip_feat,cap_clip_feat,torch.ones(cap_clip_feat.shape[0]).cuda())
        # ret.update({'CLIP Loss': ( result_loss1 + result_loss2 )* 1.0})

        return ret

