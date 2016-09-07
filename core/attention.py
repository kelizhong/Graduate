# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-6
# Time: 上午9:20
# Author: Zhu Danxiang
#

import mxnet as mx
from collections import namedtuple

SoftAttnParam = namedtuple('SoftAttnParam', ['so_proj_h_weight', 'so_alpha_weight', 'so_alpha_bias'])
LocalAttnParam = namedtuple('LocalAttnParam', ['lo_proj_h_weight'])


def make_soft_attention_alpha(proj_ctx, hidden, context_shape, param):
    proj_h = mx.sym.FullyConnected(data=hidden, num_hidden=context_shape[1], weight=param.so_proj_h_weight,
                                   no_bias=True)
    pls_ctx = mx.sym.broadcast_plus(proj_ctx, mx.sym.Reshape(data=proj_h, shape=(-1, 1, context_shape[1])))
    pls_ctx = mx.sym.Activation(data=pls_ctx, act_type='tanh')
    alpha = mx.sym.FullyConnected(data=mx.sym.Reshape(data=pls_ctx, shape=(-1, context_shape[1])), num_hidden=1,
                                  weight=param.so_alpha_weight, bias=param.so_alpha_bias)
    alpha = mx.sym.SoftmaxActivation(data=mx.sym.Reshape(data=alpha, shape=(-1, context_shape[0])))
    alpha = mx.sym.Reshape(data=alpha, shape=(-1, context_shape[0], 1))

    return alpha


def make_local_attention_alpha(context, hidden, context_shape, param):
    proj_h = mx.sym.FullyConnected(data=hidden, num_hidden=context_shape[1], weight=param.lo_proj_h_weight,
                                   no_bias=True)
    alpha = mx.sym.batch_dot(context, mx.sym.Reshape(data=proj_h, shape=(-1, context_shape[1], 1)))
    alpha = mx.sym.SoftmaxActivation(data=mx.sym.Reshape(data=alpha, shape=(-1, context_shape[0])))
    alpha = mx.sym.Reshape(data=alpha, shape=(-1, 1, context_shape[0]))

    return alpha
