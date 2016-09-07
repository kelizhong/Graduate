# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-30
# Time: 上午11:18
# Author: Zhu Danxiang
#

import mxnet as mx
from collections import namedtuple

LSTMState = namedtuple('LSTMState', ['c', 'h'])
LSTMParam = namedtuple('LSTMParam', ['i2h_weight', 'i2h_bias', 'h2h_weight', 'h2h_bias'])
LSTMAttnParam = namedtuple('LSTMAttnParam',
                           ['c2h_weight', 'c2h_bias', 'i2h_weight', 'i2h_bias', 'h2h_weight', 'h2h_bias'])


def lstm(data, num_hidden, seqidx, layeridx, param, prev_state, dropout=0.):
    if dropout > 0:
        data = mx.sym.Dropout(data=data, p=dropout)
    i2h = mx.sym.FullyConnected(data=data, num_hidden=4 * num_hidden, weight=param.i2h_weight, bias=param.i2h_bias,
                                name='t%d_l%d_i2h' % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h, num_hidden=4 * num_hidden, weight=param.h2h_weight,
                                bias=param.h2h_bias, name='t%d_l%d_h2h' % (seqidx, layeridx))
    gate = i2h + h2h
    slice_gate = mx.sym.SliceChannel(data=gate, num_outputs=4, axis=1, name='t%d_l%d_slice' % (seqidx, layeridx))
    input_gate = mx.sym.Activation(data=slice_gate[0], act_type='sigmoid')
    output_gate = mx.sym.Activation(data=slice_gate[1], act_type='sigmoid')
    forget_gate = mx.sym.Activation(data=slice_gate[2], act_type='sigmoid')
    in_transform = mx.sym.Activation(data=slice_gate[3], act_type='tanh')

    next_c = input_gate * in_transform + forget_gate * prev_state.c
    next_h = output_gate * mx.sym.Activation(data=next_c, act_type='tanh')

    return LSTMState(next_c, next_h)


def lstm_attn(data, context, num_hidden, seqidx, layeridx, param, prev_state, dropout=0.):
    if dropout:
        data = mx.sym.Dropout(data=data, p=dropout)
    i2h = mx.sym.FullyConnected(data=data, num_hidden=4 * num_hidden, weight=param.i2h_weight, bias=param.i2h_bias,
                                name='t%d_l%d_i2h' % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h, num_hidden=4 * num_hidden, weight=param.h2h_weight,
                                bias=param.h2h_bias, name='t%d_l%d_h2h' % (seqidx, layeridx))
    c2h = mx.sym.FullyConnected(data=context, num_hidden=4 * num_hidden, weight=param.c2h_weight, bias=param.c2h_bias,
                                name='t%d_l%d_c2h' % (seqidx, layeridx))
    gate = i2h + h2h + c2h
    slice_gate = mx.sym.SliceChannel(data=gate, num_outputs=4, axis=1, name='t%d_l%d_slice' % (seqidx, layeridx))
    input_gate = mx.sym.Activation(data=slice_gate[0], act_type='sigmoid')
    output_gate = mx.sym.Activation(data=slice_gate[1], act_type='sigmoid')
    forget_gate = mx.sym.Activation(data=slice_gate[2], act_type='sigmoid')
    in_transform = mx.sym.Activation(data=slice_gate[3], act_type='tanh')

    next_c = input_gate * in_transform + forget_gate * prev_state.c
    next_h = output_gate * mx.sym.Activation(data=next_c, act_type='tanh')

    return LSTMState(next_c, next_h)
