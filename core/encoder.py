# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-30
# Time: 下午7:01
# Author: Zhu Danxiang
#

import mxnet as mx
from lstm import LSTMParam, LSTMState, lstm


def make_sequence_encoder(encoder_input, seqlen, num_layer, num_hidden, dropout=0., vocab_size=0, num_embed=0,
                          with_embedding=False):
    param_cells = []
    last_state = []
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('en_l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('en_l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('en_l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('en_l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        encoder_embed_weight = mx.sym.Variable('en_embed_weight')
        encoder_input = mx.sym.Embedding(data=encoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=encoder_embed_weight, name='encoder_embed')
    slice_input = mx.sym.SliceChannel(data=encoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

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

    return last_state


def make_sequence_encoder_inference(encoder_input, seqlen, num_layer, num_hidden, dropout=0., vocab_size=0, num_embed=0,
                                    with_embedding=False):
    param_cells = []
    last_state = []
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('en_l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('en_l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('en_l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('en_l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        encoder_embed_weight = mx.sym.Variable('en_embed_weight')
        encoder_input = mx.sym.Embedding(data=encoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=encoder_embed_weight, name='encoder_embed')
    slice_input = mx.sym.SliceChannel(data=encoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

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

    output = []
    for state in last_state:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)


def make_bisequence_encoder(encoder_input, seqlen, num_layer, num_hidden, dropout=0., vocab_size=0, num_embed=0,
                            with_embedding=False):
    forward_param_cells = []
    forward_last_state = []
    backward_param_cells = []
    backward_last_state = []
    for i in xrange(num_layer):
        forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('fwd_en_l%d_i2h_weight' % i),
                                             i2h_bias=mx.sym.Variable('fwd_en_l%d_i2h_bias' % i),
                                             h2h_weight=mx.sym.Variable('fwd_en_l%d_h2h_weight' % i),
                                             h2h_bias=mx.sym.Variable('fwd_en_l%d_h2h_bias' % i)))
        forward_last_state.append(LSTMState(c=mx.sym.Variable('fwd_l%d_init_c' % i),
                                            h=mx.sym.Variable('fwd_l%d_init_h' % i)))
        backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('bwd_en_l%d_i2h_weight' % i),
                                              i2h_bias=mx.sym.Variable('bwd_en_l%d_i2h_bias' % i),
                                              h2h_weight=mx.sym.Variable('bwd_en_l%d_h2h_weight' % i),
                                              h2h_bias=mx.sym.Variable('bwd_en_l%d_h2h_bias' % i)))
        backward_last_state.append(LSTMState(c=mx.sym.Variable('bwd_l%d_init_c' % i),
                                             h=mx.sym.Variable('bwd_l%d_init_h' % i)))
    assert len(forward_last_state) == num_layer == len(backward_last_state)

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        encoder_embed_weight = mx.sym.Variable('en_embed_weight')
        encoder_input = mx.sym.Embedding(data=encoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=encoder_embed_weight, name='encoder_embed')
    slice_input = mx.sym.SliceChannel(data=encoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

    for seqidx in xrange(seqlen):
        hidden = slice_input[seqidx]

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                              param=forward_param_cells[i], prev_state=forward_last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            forward_last_state[i] = next_state

    for seqidx in xrange(seqlen - 1, -1, -1):
        hidden = slice_input[seqidx]

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                              param=backward_param_cells[i], prev_state=backward_last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            backward_last_state[i] = next_state

    last_state = []
    for i in xrange(num_layer):
        fwd_state = forward_last_state[i]
        bwd_state = backward_last_state[i]
        combine_c = fwd_state.c + bwd_state.c
        combine_h = fwd_state.h + bwd_state.h
        last_state.append(LSTMState(c=combine_c, h=combine_h))

    return last_state


def make_bisequence_encoder_inference(encoder_input, seqlen, num_layer, num_hidden, dropout=0., vocab_size=0,
                                      num_embed=0, with_embedding=False):
    forward_param_cells = []
    forward_last_state = []
    backward_param_cells = []
    backward_last_state = []
    for i in xrange(num_layer):
        forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('fwd_en_l%d_i2h_weight' % i),
                                             i2h_bias=mx.sym.Variable('fwd_en_l%d_i2h_bias' % i),
                                             h2h_weight=mx.sym.Variable('fwd_en_l%d_h2h_weight' % i),
                                             h2h_bias=mx.sym.Variable('fwd_en_l%d_h2h_bias' % i)))
        forward_last_state.append(LSTMState(c=mx.sym.Variable('fwd_l%d_init_c' % i),
                                            h=mx.sym.Variable('fwd_l%d_init_h' % i)))
        backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('bwd_en_l%d_i2h_weight' % i),
                                              i2h_bias=mx.sym.Variable('bwd_en_l%d_i2h_bias' % i),
                                              h2h_weight=mx.sym.Variable('bwd_en_l%d_h2h_weight' % i),
                                              h2h_bias=mx.sym.Variable('bwd_en_l%d_h2h_bias' % i)))
        backward_last_state.append(LSTMState(c=mx.sym.Variable('bwd_l%d_init_c' % i),
                                             h=mx.sym.Variable('bwd_l%d_init_h' % i)))
    assert len(forward_last_state) == num_layer == len(backward_last_state)

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        encoder_embed_weight = mx.sym.Variable('en_embed_weight')
        encoder_input = mx.sym.Embedding(data=encoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=encoder_embed_weight, name='encoder_embed')
    slice_input = mx.sym.SliceChannel(data=encoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)

    for seqidx in xrange(seqlen):
        hidden = slice_input[seqidx]

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                              param=forward_param_cells[i], prev_state=forward_last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            forward_last_state[i] = next_state

    for seqidx in xrange(seqlen - 1, -1, -1):
        hidden = slice_input[seqidx]

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                              param=backward_param_cells[i], prev_state=backward_last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            backward_last_state[i] = next_state

    last_state = []
    for i in xrange(num_layer):
        fwd_state = forward_last_state[i]
        bwd_state = backward_last_state[i]
        last_state.append(LSTMState(c=fwd_state.c + bwd_state.c, h=fwd_state.h + bwd_state.h))

    output = []
    for state in last_state:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)


def make_mean_encoder(encoder_input, seqlen, num_layer, num_hidden, dropout=0., vocab_size=0, num_embed=0,
                      with_embedding=False):
    param_cells = []
    last_state = []
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('en_l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('en_l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('en_l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('en_l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        encoder_embed_weight = mx.sym.Variable('en_embed_weight')
        encoder_input = mx.sym.Embedding(data=encoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=encoder_embed_weight, name='encoder_embed')
    mean_input = mx.sym.sum(data=encoder_input, axis=1, keepdims=False) / seqlen

    hidden = mean_input
    for i in xrange(num_layer):
        if i == 0:
            dp_ratio = 0
        else:
            dp_ratio = dropout
        next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=0, layeridx=i, param=param_cells[i],
                          prev_state=last_state[i], dropout=dp_ratio)
        hidden = next_state.h
        last_state[i] = next_state

    return last_state


def make_mean_encoder_inference(encoder_input, seqlen, num_layer, num_hidden, dropout=0., vocab_size=0, num_embed=0,
                                with_embedding=False):
    param_cells = []
    last_state = []
    for i in xrange(num_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('en_l%d_i2h_weight' % i),
                                     i2h_bias=mx.sym.Variable('en_l%d_i2h_bias' % i),
                                     h2h_weight=mx.sym.Variable('en_l%d_h2h_weight' % i),
                                     h2h_bias=mx.sym.Variable('en_l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    if with_embedding is True:
        assert vocab_size > 0 and num_embed > 0
        encoder_embed_weight = mx.sym.Variable('en_embed_weight')
        encoder_input = mx.sym.Embedding(data=encoder_input, input_dim=vocab_size, output_dim=num_embed,
                                         weight=encoder_embed_weight, name='encoder_embed')
    mean_input = mx.sym.sum(data=encoder_input, axis=1, keepdims=False) / seqlen

    hidden = mean_input
    for i in xrange(num_layer):
        if i == 0:
            dp_ratio = 0
        else:
            dp_ratio = dropout
        next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=0, layeridx=i, param=param_cells[i],
                          prev_state=last_state[i], dropout=dp_ratio)
        hidden = next_state.h
        last_state[i] = next_state

    output = []
    for state in last_state:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
