# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import pad_tensor, dict_to_cuda
from ..predictor import build_v_predictor
from .base_enc_dec_e2e import BaseEncoderDecoderE2E
from .build import META_ARCH_REGISTRY

__all__ = ["TransformerE2EEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class TransformerE2EEncoderDecoder(BaseEncoderDecoderE2E):
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
        v_predictor,
        swin_pth,
        freezen,
        freezen_list
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            swin_pth=swin_pth,
            freezen=freezen,
            freezen_list=freezen_list
        )
        self.v_predictor = v_predictor

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if cfg.MODEL.BERT.V_TARGET_SIZE > 0:
            v_predictor = build_v_predictor(cfg)
        else:
            v_predictor = None
        
        ret.update({ "v_predictor": v_predictor })
        return ret

    def get_extended_attention_mask(self, batched_inputs):
        vmasks = batched_inputs[kfg.ATT_MASKS]
        vmasks = vmasks.to(dtype=next(self.parameters()).dtype)
        vmasks = vmasks.unsqueeze(1).unsqueeze(2)
        ext_vmasks = (1.0 - vmasks) * -10000.0

        return {
            kfg.ATT_MASKS: vmasks,
            kfg.EXT_ATT_MASKS: ext_vmasks
        }

    def get_extended_attention_mask_clip(self, batched_inputs):
        if kfg.TOKENS_MASKS not in batched_inputs:
            batched_inputs[kfg.TOKENS_MASKS] = torch.ones((batched_inputs[kfg.ATT_MASKS].size(0), self.max_seq_len-1)).cuda()

        tmasks = batched_inputs[kfg.TOKENS_MASKS]
        seq_length = tmasks.size(-1)
        tmasks = tmasks.to(dtype=next(self.parameters()).dtype)
        # ext_u_tmasks = tmasks.unsqueeze(1).unsqueeze(2)
        # ext_u_tmasks = (1.0 - ext_u_tmasks) * -10000.0

        ext_g_tmasks = torch.tril(torch.ones(
            (seq_length, seq_length), dtype=tmasks.dtype, device=tmasks.device))
        ext_g_tmasks = ext_g_tmasks.unsqueeze(0).expand(
            (tmasks.size(0), seq_length, seq_length))
        ext_g_tmasks = ext_g_tmasks * tmasks.unsqueeze(1)
        ext_g_tmasks = ext_g_tmasks.to(dtype=next(self.parameters()).dtype)

        clip_size = batched_inputs['CLIP_SIZE']
        # clip_size_tmasks = torch.ones((tmasks.size(0), clip_size+seq_length, clip_size), dtype=tmasks.dtype, device=tmasks.device)
        # if self.training:
        #     mask_rate=0.1
        #     clip_size_value = torch.rand((tmasks.size(0), clip_size + seq_length, clip_size), dtype=tmasks.dtype,device=tmasks.device)
        #     clip_size_tmasks = (clip_size_value > mask_rate).int()
        # else:
        #     clip_size_tmasks = torch.ones((tmasks.size(0), clip_size + seq_length, clip_size), dtype=tmasks.dtype,device=tmasks.device)

        clip_size_tmasks = torch.ones((tmasks.size(0), clip_size + seq_length, clip_size), dtype=tmasks.dtype,
                                      device=tmasks.device)
        zeros_tmasks = torch.zeros((tmasks.size(0),clip_size, seq_length ), dtype=tmasks.dtype, device=tmasks.device)
        midd = torch.cat([zeros_tmasks, ext_g_tmasks], dim=1)
        final = torch.cat([clip_size_tmasks, midd], dim=2)


        final = final.unsqueeze(1)
        final = (1.0 - final) * -10000.0

        return {
            kfg.TOKENS_MASKS: tmasks,
            # kfg.EXT_U_TOKENS_MASKS: ext_u_tmasks,
            kfg.EXT_G_TOKENS_MASKS: final,
        }

    def _forward(self, batched_inputs):
        inputs = batched_inputs

        if kfg.IMG_INPUT or 'IMG_PATH' in batched_inputs:
            backbone_out = self.backbone(batched_inputs)
            inputs.update(backbone_out)

        vfeats, vmasks = pad_tensor(inputs[kfg.ATT_FEATS], padding_value=0, use_mask=True)
        inputs.update({kfg.ATT_MASKS: vmasks.cuda(), kfg.GLOBAL_FEATS:vfeats.mean(-2)})

        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(inputs)
        inputs.update(ve_out)

        encoder_out_v = self.encoder(inputs, mode='v')
        inputs.update(encoder_out_v)

        clip_out = self.backbone(batched_inputs, mode='clip')
        inputs.update(clip_out)

        te_out = self.token_embed(batched_inputs)
        inputs.update(te_out)

        masks = self.get_extended_attention_mask_clip(batched_inputs)
        inputs.update(masks)

        decoder_out = self.decoder(inputs)
        inputs.update(decoder_out)

        tlogits = self.predictor(inputs)
        inputs.update(tlogits)

        return inputs