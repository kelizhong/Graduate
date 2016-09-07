# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-30
# Time: 上午10:15
# Author: Zhu Danxiang
#

import h5py
import mxnet as mx
import numpy as np


class CaptionBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data_names = data_names
        self.data = data
        self.label_names = label_names
        self.label = label

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class CaptionIter(mx.io.DataIter):
    def __init__(self, filename, init_state, batch_size):
        super(CaptionIter, self).__init__()
        self.f = h5py.File(filename, 'r')
        self.feat_shape = self.f.attrs['feat_shape']
        self.num_image = self.f.attrs['num_image']
        self.num_caption = self.f.attrs['num_caption']
        self.max_length = self.f.attrs['max_length'] - 1
        self.batch_size = batch_size
        self.init_state = init_state
        self.init_state_array = [mx.nd.zeros(x[1]) for x in self.init_state]
        self.batch_targ_len = []
        self.batch_start_ix = []
        self.batch_end_ix = []
        self.num_batch = 0

        self.provide_data = [('encoder_input', (self.batch_size, self.feat_shape[0], self.feat_shape[1])),
                             ('decoder_input', (self.batch_size, self.max_length))] + self.init_state
        self.provide_label = [('decoder_output', (self.batch_size, self.max_length))]

    def __iter__(self):
        for i in xrange(0, self.num_image, self.batch_size / 5):
            if i + self.batch_size >= self.num_image:
                break
            feat = np.repeat(self.f['feats'][i: i + self.batch_size / 5], 5, axis=0)
            caption_input = self.f['caps'][i * 5: i * 5 + self.batch_size, :-1]
            caption_output = self.f['caps'][i * 5: i * 5 + self.batch_size, 1:]

            feat = mx.nd.array(feat)
            caption_input = mx.nd.array(caption_input)
            caption_output = mx.nd.array(caption_output)

            data_all = [feat] + [caption_input] + self.init_state_array
            data_names = ['encoder_input', 'decoder_input'] + [x[0] for x in self.init_state]
            label_all = [caption_output]
            label_names = ['decoder_output']

            batch_data = CaptionBatch(data_names, data_all, label_names, label_all)
            yield batch_data
