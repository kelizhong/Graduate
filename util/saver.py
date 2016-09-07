# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 下午3:51
# Author: Zhu Danxiang
#

import os
import cPickle
import mxnet as mx


class BestScoreSaver(object):
    def __init__(self, prefix, epoch, config):
        self.dirname = os.path.dirname(prefix)
        self.prefix = prefix
        self.epoch = epoch
        self.config = config
        self.best_score = None

    def update(self, score, symbol, arg_params, aux_params):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        if not os.path.exists(self.prefix + '-config.pkl'):
            cPickle.dump(self.config, open(self.prefix + '-config.pkl', 'w'))
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            mx.model.save_checkpoint(self.prefix, self.epoch, symbol, arg_params, aux_params)
