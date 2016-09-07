# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 下午3:46
# Author: Zhu Danxiang
#

import numpy as np
from abc import abstractmethod


class Metric(object):
    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        if self.verbose is True:
            assert 'batch_size' in kwargs
            assert 'vocab_size' in kwargs
            assert 'ix2token' in kwargs
            self.batch_size = kwargs['batch_size']
            self.vocab_size = kwargs['vocab_size']
            self.ix2token = kwargs['ix2token']

    @abstractmethod
    def calculate(self, label, pred):
        pass

    def verbose_print(self, pred):
        assert self.verbose is True
        output = np.reshape(pred, (-1, self.batch_size, self.vocab_size)).transpose((1, 0, 2))
        output = np.argmax(output[np.random.randint(self.batch_size)], axis=1)
        print ' '.join([self.ix2token[x] for x in output])


class Perplexity(Metric):
    def calculate(self, label, pred):
        if self.verbose is True:
            self.verbose_print(pred)
        label = label.T.reshape((-1,))
        loss = 0.
        cnt = 0
        for i in range(pred.shape[0]):
            if label[i] == 0:
                continue
            cnt += 1
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
        return np.exp(loss / cnt)


class Accuracy(Metric):
    def calculate(self, label, pred):
        if self.verbose is True:
            self.verbose_print(pred)
        label = label.T.reshape((-1,)).astype(np.int32)
        pred = np.argmax(pred, axis=1)
        assert len(label) == len(pred)
        nElement = len(label)
        cnt = np.sum(label == pred)
        return float(cnt) / nElement


class NegativeLogLikehood(Metric):
    def calculate(self, label, pred):
        if self.verbose is True:
            self.verbose_print(pred)
        label = label.T.reshape((-1,))
        loss = 0.
        for i in range(pred.shape[0]):
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
        return loss / label.size
