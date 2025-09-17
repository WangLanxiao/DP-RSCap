# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone
"""

def build_backbone(cfg):
    backbone = BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE)(cfg) if len(cfg.MODEL.BACKBONE) > 0 else None
    return backbone

def add_backbone_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.BACKBONE) > 0:
        BACKBONE_REGISTRY.get(tmp_cfg.MODEL.BACKBONE).add_config(cfg)