# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import copy
import numpy as np
import weakref
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
import torchvision
from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor, dict_to_cuda, flat_list_of_lists
from ..embedding import build_embeddings
from ..backbone import build_backbone, add_backbone_config
from ..encoder import build_encoder, add_encoder_config
from ..decoder import build_decoder, add_decoder_config
from ..predictor import build_predictor, add_predictor_config
from ..decode_strategy import build_beam_searcher, build_greedy_decoder

class BaseEncoderDecoderE2E(nn.Module, metaclass=ABCMeta):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        backbone,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
        swin_pth,
        freezen,
        freezen_list,
    ):
        super(BaseEncoderDecoderE2E, self).__init__()
        self.token_embed = token_embed
        self.visual_embed = visual_embed
        # model = torchvision.models.resnet101(pretrained=True, progress=True)
        self.backbone=backbone
        print('load pretrained weights from ',swin_pth)
        # self.backbone.load_weights(
        #     swin_pth
        # )
        # Freeze parameters
        # if freezen:
        #     for idx, (_name, _weight) in enumerate(self.backbone.named_parameters()):
        #         for ss in freezen_list[0]:
        #             if ss in _name:
        #                 _weight.requires_grad = False


            # pd=True
            # for idx,(_name, _weight) in enumerate(self.backbone.named_parameters()):
            #     print(idx,'   ',_name)
            #     if 'layers.3' in _name:
            #         pd = False
            #     #     _weight.requires_grad = False
            #     # if _name in ['norm.weight','norm.bias']:
            #     #     _weight.requires_grad = True
            #     if pd:
            #         _weight.requires_grad = False
            #     else:
            #         # print(_name)
            #         continue


        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.greedy_decoder = greedy_decoder
        self.beam_searcher = beam_searcher
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(cls, cfg):
        return {
            "token_embed": build_embeddings(cfg, cfg.MODEL.TOKEN_EMBED.NAME),
            "visual_embed": build_embeddings(cfg, cfg.MODEL.VISUAL_EMBED.NAME),
            "backbone": build_backbone(cfg),
            "encoder": build_encoder(cfg),
            "decoder": build_decoder(cfg),
            "predictor": build_predictor(cfg),
            "greedy_decoder": build_greedy_decoder(cfg),
            "beam_searcher": build_beam_searcher(cfg),
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "swin_pth": cfg.MODEL.SWIN_PTH,
            "freezen": cfg.MODEL.FREEZEN,
            "freezen_list": cfg.SOLVER.SPECIAL_WEIGHT_DECAY
        }

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        add_backbone_config(cfg, tmp_cfg)
        add_encoder_config(cfg, tmp_cfg)
        add_decoder_config(cfg, tmp_cfg)
        add_predictor_config(cfg, tmp_cfg)

    @abstractmethod
    def get_extended_attention_mask(self, batched_inputs):
        pass

    def forward(self, batched_inputs, use_beam_search=None, output_sents=False):
        if use_beam_search is None:
            return self._forward(batched_inputs)
        elif use_beam_search == False or self.beam_searcher.beam_size == 1:
            return self.greedy_decode(batched_inputs, output_sents)
        else:
            return self.decode_beam_search(batched_inputs, output_sents)

    @abstractmethod
    def _forward(self, batched_inputs):
        pass

    def bind_or_init_weights(self):
        pass

    def preprocess_batch(self, batched_inputs):
        sample_per_sample = batched_inputs[0].get(kfg.SAMPLE_PER_SAMPLE, 1)
        if kfg.IMG_INPUT in batched_inputs[0]:
            imgfeats = [x[kfg.IMG_INPUT] for x in batched_inputs]
            if sample_per_sample > 1:
                imgfeats = flat_list_of_lists(imgfeats)
            imgfeats, vmasks = pad_tensor(imgfeats, padding_value=0, use_mask=True)
            ret = { kfg.IMG_INPUT: imgfeats }
        elif 'IMG_PATH' in batched_inputs[0]:
            ret = {}
        else:
            vfeats = [x[kfg.ATT_FEATS] for x in batched_inputs]
            pyrfeats = [x[kfg.PYR_FEATS] for x in batched_inputs]
            globfeats = [x[kfg.GLOBAL_FEATS] for x in batched_inputs]
            if sample_per_sample > 1:
                vfeats = flat_list_of_lists(vfeats)
                pyrfeats = flat_list_of_lists(pyrfeats)
                globfeats = flat_list_of_lists(globfeats)
            vfeats, vmasks = pad_tensor(vfeats, padding_value=0, use_mask=True)
            ret = {
                kfg.ATT_FEATS: vfeats,
                kfg.PYR_FEATS: pyrfeats,
                kfg.GLOBAL_FEATS: globfeats
            }

        if kfg.GLOBAL_FEATS in batched_inputs[0]:
            gv_feats = [x[kfg.GLOBAL_FEATS] for x in batched_inputs]
            gv_feats = pad_tensor(gv_feats, padding_value=0, use_mask=False) 
            ret.update( { kfg.GLOBAL_FEATS: gv_feats } )

        if kfg.TREE_M in batched_inputs[0]:
            tree_matric = [x[kfg.TREE_M] for x in batched_inputs]
            tree_matric = pad_tensor(tree_matric, padding_value=-1, use_mask=False)
            ret.update( { kfg.TREE_M: tree_matric} )

        if kfg.G_TOKENS_IDS in batched_inputs[0]:
            g_tokens_ids = [x[kfg.G_TOKENS_IDS] for x in batched_inputs]
            g_tokens_ids, tmasks = pad_tensor(g_tokens_ids, padding_value=0, use_mask=True)
            ret.update( { kfg.G_TOKENS_IDS: g_tokens_ids, kfg.TOKENS_MASKS: tmasks} )

        if kfg.G_TARGET_IDS in batched_inputs[0]:
            g_target_ids = [x[kfg.G_TARGET_IDS] for x in batched_inputs]
            g_target_ids = pad_tensor(g_target_ids, padding_value=-1, use_mask=False)
            ret.update({ kfg.G_TARGET_IDS: g_target_ids })

        if kfg.G_TOKENS_TYPE in batched_inputs[0]:
            g_tokens_type = [x[kfg.G_TOKENS_TYPE] for x in batched_inputs]
            g_tokens_type = pad_tensor(g_tokens_type, padding_value=1, use_mask=False)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })

        if kfg.IMG_INPUT in batched_inputs[0]:
            img_input = [x[kfg.IMG_INPUT] for x in batched_inputs]
            img_input = pad_tensor(img_input, padding_value=0, use_mask=False)
            ret.update({ kfg.IMG_INPUT: img_input })

        if kfg.ATT_FEATS in batched_inputs[0]:
            att_input = [x[kfg.ATT_FEATS] for x in batched_inputs]
            att_input = pad_tensor(att_input, padding_value=0, use_mask=False)
            ret.update({ kfg.ATT_FEATS: att_input })

        if kfg.PYR_FEATS in batched_inputs[0]:
            pyr_input = [x[kfg.PYR_FEATS] for x in batched_inputs]
            pyr_input = pad_tensor(pyr_input, padding_value=0, use_mask=False)
            ret.update({ kfg.PYR_FEATS: pyr_input})

        if 'CLASS_ID' in batched_inputs[0]:
            cls_input = torch.cat([torch.stack(x['CLASS_ID']) for x in batched_inputs],0)
            ret.update({ 'CLASS_ID': cls_input})


        ######################################################################################

        if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
            repeat_num = batched_inputs[0][kfg.SEQ_PER_SAMPLE].item()
            if kfg.IMG_INPUT in batched_inputs[0]:
                batch_size, channel, H_dim, W_dim = imgfeats.size()[0:4]
                imgfeats = imgfeats.unsqueeze(1).expand(batch_size, repeat_num, channel, H_dim, W_dim)
                imgfeats = imgfeats.reshape(-1, channel, H_dim, W_dim)
                ret.update({ kfg.IMG_INPUT: imgfeats})

            if kfg.GLOBAL_FEATS in batched_inputs[0]:
                batch_size, gv_feat_dim = gv_feats.size()
                gv_feats = gv_feats.view(batch_size, -1, gv_feat_dim).expand(batch_size, repeat_num, gv_feat_dim)
                gv_feats = gv_feats.reshape(-1, gv_feat_dim)
                ret.update({ kfg.GLOBAL_FEATS: gv_feats })

            if kfg.ATT_FEATS in batched_inputs[0]:
                batch_size, att_numb, att_feat_dim = att_input.size()
                att_input = att_input.view(batch_size, -1,att_numb, att_feat_dim).expand(batch_size, repeat_num,att_numb, att_feat_dim)
                att_input = att_input.reshape(-1,att_numb, att_feat_dim)
                ret.update({ kfg.ATT_FEATS: att_input })

            if kfg.PYR_FEATS in batched_inputs[0]:
                batch_size, Layer_num, pyr_numb, pyr_feat_dim = pyr_input.size()
                pyr_input = pyr_input.view(batch_size, -1, Layer_num, pyr_numb, pyr_feat_dim).expand(batch_size, repeat_num, Layer_num, pyr_numb, pyr_feat_dim)
                pyr_input = pyr_input.reshape(-1, Layer_num, pyr_numb, pyr_feat_dim)
                ret.update({kfg.PYR_FEATS: pyr_input})

        dict_to_cuda(ret)
        if kfg.IDS in batched_inputs[0]:
            ids = [x[kfg.IDS]  for x in batched_inputs ]
            if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
                ids = np.repeat(np.expand_dims(ids, axis=1), repeat_num, axis=1).flatten()
            ret.update({ kfg.IDS: ids })

        if 'ENTITY_ID' in batched_inputs[0]:
            entity_id = [x['ENTITY_ID']  for x in batched_inputs ]
            if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
                entity_id = np.repeat(np.expand_dims(entity_id, axis=1), repeat_num, axis=1).flatten()
            ret.update({'ENTITY_ID': entity_id})

        if 'ENTITY_SCORE' in batched_inputs[0]:
            entity_score = [x['ENTITY_SCORE']  for x in batched_inputs ]
            if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
                entity_score = np.repeat(np.expand_dims(entity_score, axis=1), repeat_num, axis=1).flatten()
            ret.update({'ENTITY_SCORE': entity_score})

        if 'ENTITY_NAME' in batched_inputs[0]:
            entity_name = [x['ENTITY_NAME'] for x in batched_inputs]
            if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
                entity_name = [x for x in entity_name for i in range(batched_inputs[0][kfg.SEQ_PER_SAMPLE])]
            ret.update({'ENTITY_NAME': entity_name})

        if 'IMG_PATH' in batched_inputs[0]:
            img_path_input = [x['IMG_PATH'] for x in batched_inputs]
            if kfg.SEQ_PER_SAMPLE in batched_inputs[0]:
                img_path_input = [x for x in img_path_input for i in range(batched_inputs[0][kfg.SEQ_PER_SAMPLE])]
            ret.update({ 'IMG_PATH': img_path_input})

        if kfg.SAMPLE_PER_SAMPLE in batched_inputs[0]:
            ret.update({ kfg.SAMPLE_PER_SAMPLE: sample_per_sample})

        return ret

    def greedy_decode(self, batched_inputs, output_sents=False):
        return self.greedy_decoder(
            batched_inputs, 
            output_sents,
            model=weakref.proxy(self)
        )

    def decode_beam_search(self, batched_inputs, output_sents=False):
        return self.beam_searcher(
            batched_inputs, 
            output_sents,
            model=weakref.proxy(self)
        )
