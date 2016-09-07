# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 下午3:31
# Author: Zhu Danxiang
#

from bleu_scorer import BleuScorer
from cider_scorer import CiderScorer
from rouge import Rouge
from collections import OrderedDict

scorer_type = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'ROUGE', 'CIDEr']


def compute_score(hypos, refs):
    assert type(hypos) is list
    assert type(refs) is list
    scorers = [(BleuScorer(n=4), ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']),
               (Rouge(), ['ROUGE']),
               (CiderScorer(), ['CIDEr'])]

    score = OrderedDict()
    for scorer, method in scorers:
        for hypo, ref in zip(hypos, refs):
            scorer += (hypo[0], ref)
        value, _ = scorer.compute_score()

        if len(method) > 1:
            for m, v in zip(method, value):
                score[m] = v
        else:
            score[method[0]] = value

    return score
