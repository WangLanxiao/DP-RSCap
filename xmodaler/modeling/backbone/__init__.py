# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""


from .build import build_backbone, add_backbone_config
from .clip import CLIP

__all__ = list(globals().keys())
