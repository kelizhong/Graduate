# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-30
# Time: 下午8:47
# Author: Zhu Danxiang
#

import mxnet as mx
from lstm import LSTMParam, LSTMState, lstm, LSTMAttnParam, lstm_attn
from attention import SoftAttnParam, LocalAttnParam
from attention import make_soft_attention_alpha, make_local_attention_alpha


def make_sequence_decoder(decoder_input, decoder_output, init_state, seqlen, num_layer, num_hidden, num_label,
                          dropout=0., vocab_size=0, num_embed=0, with_embedding=False):
    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    param_cells = []
    last_state = init_state
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('de_l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('de_l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('de_l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('de_l%d_h2h_bias' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        decoder_embed_weight = mx.sym.Variable('de_embed_weight')
        decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=decoder_embed_weight, name='decoder_embed')
    slice_input = mx.sym.SliceChannel(data=decoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

    hidden_all = []
    for seqidx in xrange(seqlen):
        hidden = slice_input[seqidx]

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i, param=param_cells[i],
                              prev_state=last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            last_state[i] = next_state
        hidden = mx.sym.Concat(*[hidden, slice_input[seqidx]], dim=1)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label, weight=fc_weight, bias=fc_bias,
                               name='fc')

    decoder_output = mx.sym.transpose(data=decoder_output)
    decoder_output = mx.sym.Reshape(data=decoder_output, shape=(-1,))

    sm = mx.sym.SoftmaxOutput(data=fc, label=decoder_output, name='decoder_softmax')
    return sm


def make_sequence_decoder_inference(decoder_input, num_layer, num_hidden, num_label, dropout=0., vocab_size=0,
                                    num_embed=0, with_embedding=False):
    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    param_cells = []
    last_state = []
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('de_l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('de_l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('de_l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('de_l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        decoder_embed_weight = mx.sym.Variable('de_embed_weight')
        decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=decoder_embed_weight, name='decoder_embed')
    hidden = decoder_input
    for i in xrange(num_layer):
        if i == 0:
            dp_ratio = 0
        else:
            dp_ratio = dropout
        next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=0, layeridx=i, param=param_cells[i],
                          prev_state=last_state[i], dropout=dp_ratio)
        hidden = next_state.h
        last_state[i] = next_state
    if dropout > 0:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)
    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label, weight=fc_weight, bias=fc_bias, name='fc')
    sm = mx.sym.SoftmaxActivation(data=fc, name='decoder_softmax')

    output = [sm]
    for state in last_state:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)


def make_soft_attention_decoder(decoder_input, decoder_output, init_state, context, context_shape, seqlen, num_layer,
                                num_hidden, num_label, dropout=0., vocab_size=0, num_embed=0, with_embedding=False):
    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    proj_ctx_weight = mx.sym.Variable('proj_ctx_weight')
    proj_ctx_bias = mx.sym.Variable('proj_ctx_bias')
    so_attn_param = SoftAttnParam(so_proj_h_weight=mx.sym.Variable('so_proj_h_weight'),
                                  so_alpha_weight=mx.sym.Variable('so_alpha_weight'),
                                  so_alpha_bias=mx.sym.Variable('so_alpha_bias'))
    lstm_param_cells = []
    last_state = init_state
    for i in xrange(num_layer):
        if i == 0:
            lstm_param_cells.append(LSTMAttnParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                                  i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                                  h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                                  h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i),
                                                  c2h_weight=mx.sym.Variable('l%d_c2h_weight' % i),
                                                  c2h_bias=mx.sym.Variable('l%d_c2h_bias' % i)))
        else:
            lstm_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                              i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                              h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                              h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
    assert len(last_state) == num_layer

    proj_ctx = mx.sym.FullyConnected(data=mx.sym.Reshape(data=context, shape=(-1, context_shape[1])),
                                     num_hidden=context_shape[1], weight=proj_ctx_weight, bias=proj_ctx_bias)
    proj_ctx = mx.sym.Reshape(data=proj_ctx, shape=(-1, context_shape[0], context_shape[1]))
    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        decoder_embed_weight = mx.sym.Variable('de_embed_weight')
        decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=decoder_embed_weight, name='decoder_embed')
    slice_input = mx.sym.SliceChannel(data=decoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

    hidden_all = []
    for seqidx in xrange(seqlen):
        hidden = slice_input[seqidx]
        alpha = make_soft_attention_alpha(proj_ctx=proj_ctx, hidden=hidden, context_shape=context_shape,
                                          param=so_attn_param)
        weighted_context = mx.sym.broadcast_mul(context, alpha)
        weighted_context = mx.sym.sum(data=weighted_context, axis=1)
        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
                next_state = lstm_attn(data=hidden, context=weighted_context, num_hidden=num_hidden, seqidx=seqidx,
                                       layeridx=i, param=lstm_param_cells[i], prev_state=last_state[i],
                                       dropout=dp_ratio)
            else:
                dp_ratio = dropout
                next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                                  param=lstm_param_cells[i],
                                  prev_state=last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            last_state[i] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label, weight=fc_weight, bias=fc_bias,
                               name='fc')
    decoder_output = mx.sym.transpose(data=decoder_output)
    decoder_output = mx.sym.Reshape(data=decoder_output, shape=(-1,))

    sm = mx.sym.SoftmaxOutput(data=fc, label=decoder_output, name='decoder_softmax')
    return sm


def make_local_attention_decoder(decoder_input, decoder_output, init_state, context, context_shape, seqlen, num_layer,
                                 num_hidden, num_label, dropout=0., vocab_size=0, num_embed=0, with_embedding=False):
    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    lo_attn_param = LocalAttnParam('lo_proj_h_weight')
    lo_concat_weight = mx.sym.Variable('lo_concat_weight')
    lstm_param_cells = []
    last_state = init_state
    for i in xrange(num_layer):
        if i == 0:
            lstm_param_cells.append(LSTMAttnParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                                  i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                                  h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                                  h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i),
                                                  c2h_weight=mx.sym.Variable('l%d_c2h_weight' % i),
                                                  c2h_bias=mx.sym.Variable('l%d_c2h_bias' % i)))
        else:
            lstm_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                              i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                              h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                              h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        decoder_embed_weight = mx.sym.Variable('de_embed_weight')
        decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=decoder_embed_weight, name='decoder_embed')
    slice_input = mx.sym.SliceChannel(data=decoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

    hidden_all = []
    for seqidx in xrange(seqlen):
        hidden = slice_input[seqidx]

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                              param=lstm_param_cells[i],
                              prev_state=last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            last_state[i] = next_state

        weight_context = make_local_attention_alpha(context, hidden, context_shape, lo_attn_param)
        hidden = mx.sym.Concat(*[weight_context, hidden], axis=1)
        hidden = mx.sym.FullyConnected(data=hidden, num_hidden=num_hidden, weight=lo_concat_weight, no_bias=True)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label, weight=fc_weight, bias=fc_bias,
                               name='fc')
    decoder_output = mx.sym.transpose(data=decoder_output)
    decoder_output = mx.sym.Reshape(data=decoder_output, shape=(-1,))

    sm = mx.sym.SoftmaxOutput(data=fc, label=decoder_output, name='decoder_softmax')
    return sm
