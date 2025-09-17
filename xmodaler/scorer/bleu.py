#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .build import SCORER_REGISTRY
from .bleu_scorer import BleuScorer
import numpy as np


__all__ = ['Bleu']

@SCORER_REGISTRY.register()
class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(len(gts) == len(res))
        imgIds = list(range(len(gts)))

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            # hypo = res[id]
            hypo = [' '.join(map(str,res[id]))]
            ref = [' '.join(map(str,j)) for j in gts[id]]
            # ref = [gts[id]

            # Sanity check.
            # assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            # assert(type(ref) is list)
            # assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return np.array(sum(score)), np.array([(scores[0][idx]+scores[1][idx]+scores[2][idx]+scores[3][idx])/4 for idx in range(len(scores[0]))])

    def method(self):
        return "Bleu"
