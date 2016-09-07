# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-30
# Time: 下午9:18
# Author: Zhu Danxiang
#

import mxnet as mx
from lstm import LSTMParam, LSTMState, LSTMAttnParam, lstm, lstm_attn
from encoder import make_sequence_encoder, make_bisequence_encoder, make_mean_encoder
from decoder import make_sequence_decoder, make_soft_attention_decoder
from attention import make_soft_attention_alpha, SoftAttnParam, make_local_attention_alpha, LocalAttnParam


def make_show_and_tell_symbol(seqlen, num_layer, num_hidden, num_label, vocab_size, num_embed, dropout=0.):
    encoder_input = mx.sym.Variable('encoder_input')
    decoder_input = mx.sym.Variable('decoder_input')
    decoder_output = mx.sym.Variable('decoder_output')
    encoder_last_state = make_mean_encoder(encoder_input=encoder_input, seqlen=1, num_layer=num_layer,
                                           num_hidden=num_hidden, dropout=dropout)
    sm = make_sequence_decoder(decoder_input=decoder_input, decoder_output=decoder_output,
                               init_state=encoder_last_state, seqlen=seqlen, num_layer=num_layer, num_hidden=num_hidden,
                               num_label=num_label, dropout=dropout, vocab_size=vocab_size, num_embed=num_embed,
                               with_embedding=True)
    return sm


def make_show_attend_and_tell_symbol(context_shape, seqlen, num_layer, num_hidden, num_label, vocab_size, num_embed,
                                     dropout=0.):
    encoder_input = mx.sym.Variable('encoder_input')
    decoder_input = mx.sym.Variable('decoder_input')
    decoder_output = mx.sym.Variable('decoder_output')
    encoder_last_state = make_mean_encoder(encoder_input=encoder_input, seqlen=context_shape[0], num_layer=num_layer,
                                           num_hidden=num_hidden, dropout=dropout)
    sm = make_soft_attention_decoder(decoder_input=decoder_input, decoder_output=decoder_output,
                                     init_state=encoder_last_state, context=encoder_input, context_shape=context_shape,
                                     seqlen=seqlen, num_layer=num_layer, num_hidden=num_hidden, num_label=num_label,
                                     dropout=dropout, vocab_size=vocab_size, num_embed=num_embed, with_embedding=True)
    return sm


def make_strict_show_attend_and_tell_symbol(context_shape, seqlen, num_layer, num_hidden, num_label, vocab_size,
                                            num_embed, dropout=0.):
    encoder_input = mx.sym.Variable('encoder_input')
    decoder_input = mx.sym.Variable('decoder_input')
    decoder_output = mx.sym.Variable('decoder_output')
    mean_input = mx.sym.sum(data=encoder_input, axis=1, keepdims=False) / context_shape[0]
    context = encoder_input

    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    decoder_embed_weight = mx.sym.Variable('de_embed_weight')
    proj_ctx_weight = mx.sym.Variable('proj_ctx_weight')
    proj_ctx_bias = mx.sym.Variable('proj_ctx_bias')
    # doubly_weight = mx.sym.Variable('doubly_weight')
    # doubly_bias = mx.sym.Variable('doubly_bias')
    so_attn_param = SoftAttnParam(so_proj_h_weight=mx.sym.Variable('so_proj_h_weight'),
                                  so_alpha_weight=mx.sym.Variable('so_alpha_weight'),
                                  so_alpha_bias=mx.sym.Variable('so_alpha_bias'))
    lstm_param_cells = []
    last_state = []
    for i in xrange(num_layer):
        lstm_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                          i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                          h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                          h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
        last_state.append(
            LSTMState(c=mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_c' % i),
                      h=mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_h' % i)))
    assert len(last_state) == num_layer

    proj_ctx = mx.sym.FullyConnected(data=mx.sym.Reshape(data=context, shape=(-1, context_shape[1])),
                                     num_hidden=context_shape[1], weight=proj_ctx_weight, bias=proj_ctx_bias)
    proj_ctx = mx.sym.Reshape(data=proj_ctx, shape=(-1, context_shape[0], context_shape[1]))
    decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                     weight=decoder_embed_weight, name='decoder_embed')
    slice_input = mx.sym.SliceChannel(data=decoder_input, num_outputs=seqlen, axis=1, squeeze_axis=1)
    # weighted_context = mx.sym.sum(context, axis=1) / context_shape[0]
    ctx_tran_weight = mx.sym.Variable('ctx_tran_weight')

    hidden_all = []
    for seqidx in xrange(seqlen):
        hidden = slice_input[seqidx]
        # hidden_concat = mx.sym.Concat(*[hidden, last_state[-1].h, weighted_context], dim=1)
        alpha = make_soft_attention_alpha(proj_ctx=proj_ctx, hidden=last_state[-1].h, context_shape=context_shape,
                                          param=so_attn_param)
        weighted_context = mx.sym.broadcast_mul(context, alpha)
        weighted_context = mx.sym.sum(data=weighted_context, axis=1)
        # beta = mx.sym.FullyConnected(data=hidden, num_hidden=1, weight=doubly_weight, bias=doubly_bias,
        #                              name='doubly_attn')
        # beta = mx.sym.Activation(data=beta, act_type='sigmoid')
        # weighted_context = mx.sym.broadcast_mul(weighted_context, mx.sym.Reshape(data=beta, shape=(-1, 1)))
        hidden = mx.sym.Concat(*[hidden, weighted_context], dim=1)

        for i in xrange(num_layer):
            if i == 0:
                dp_ratio = 0
            else:
                dp_ratio = dropout
            next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=seqidx, layeridx=i,
                              param=lstm_param_cells[i], prev_state=last_state[i], dropout=dp_ratio)
            hidden = next_state.h
            last_state[i] = next_state
        # weighted_context = mx.sym.FullyConnected(data=weighted_context, num_hidden=num_hidden, weight=ctx_tran_weight,
        #                                          no_bias=True)
        # hidden = hidden + weighted_context
        # hidden = mx.sym.Concat(*[hidden, slice_input[seqidx]], dim=1)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label, weight=fc_weight, bias=fc_bias, name='fc')
    decoder_output = mx.sym.transpose(data=decoder_output)
    decoder_output = mx.sym.Reshape(data=decoder_output, shape=(-1,))

    sm = mx.sym.SoftmaxOutput(data=fc, label=decoder_output, name='decoder_softmax')
    return sm


def make_strict_show_attend_and_tell_inference(context_shape, num_layer, num_hidden, num_label, vocab_size,
                                               num_embed, dropout=0., for_init=True):
    encoder_input = mx.sym.Variable('encoder_input')
    if for_init is True:
        mean_input = mx.sym.sum(data=encoder_input, axis=1, keepdims=False) / context_shape[0]
        output = []
        for i in xrange(num_layer):
            output.append(mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_c' % i))
            output.append(mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_h' % i))
        return mx.sym.Group(output)

    decoder_input = mx.sym.Variable('decoder_input')
    context = encoder_input

    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    decoder_embed_weight = mx.sym.Variable('de_embed_weight')
    proj_ctx_weight = mx.sym.Variable('proj_ctx_weight')
    proj_ctx_bias = mx.sym.Variable('proj_ctx_bias')
    doubly_weight = mx.sym.Variable('doubly_weight')
    doubly_bias = mx.sym.Variable('doubly_bias')
    so_attn_param = SoftAttnParam(so_proj_h_weight=mx.sym.Variable('so_proj_h_weight'),
                                  so_alpha_weight=mx.sym.Variable('so_alpha_weight'),
                                  so_alpha_bias=mx.sym.Variable('so_alpha_bias'))
    lstm_param_cells = []
    last_state = []
    for i in xrange(num_layer):
        lstm_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                          i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                          h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                          h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    proj_ctx = mx.sym.FullyConnected(data=mx.sym.Reshape(data=context, shape=(-1, context_shape[1])),
                                     num_hidden=context_shape[1], weight=proj_ctx_weight, bias=proj_ctx_bias)
    proj_ctx = mx.sym.Reshape(data=proj_ctx, shape=(-1, context_shape[0], context_shape[1]))
    decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                     weight=decoder_embed_weight, name='decoder_embed')
    ctx_tran_weight = mx.sym.Variable('ctx_tran_weight')
    hidden = decoder_input
    alpha = make_soft_attention_alpha(proj_ctx=proj_ctx, hidden=last_state[-1].h, context_shape=context_shape,
                                      param=so_attn_param)
    weighted_context = mx.sym.broadcast_mul(context, alpha)
    weighted_context = mx.sym.sum(data=weighted_context, axis=1)
    beta = mx.sym.FullyConnected(data=hidden, num_hidden=1, weight=doubly_weight, bias=doubly_bias,
                                 name='doubly_attn')
    beta = mx.sym.Activation(data=beta, act_type='sigmoid')
    weighted_context = mx.sym.broadcast_mul(weighted_context, mx.sym.Reshape(data=beta, shape=(-1, 1)))
    hidden = mx.sym.Concat(*[hidden, weighted_context], dim=1)

    for i in xrange(num_layer):
        if i == 0:
            dp_ratio = 0
        else:
            dp_ratio = dropout
        next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=0, layeridx=i,
                          param=lstm_param_cells[i], prev_state=last_state[i], dropout=dp_ratio)
        hidden = next_state.h
        last_state[i] = next_state
    weighted_context = mx.sym.FullyConnected(data=weighted_context, num_hidden=num_hidden, weight=ctx_tran_weight,
                                             no_bias=True)
    hidden = hidden + weighted_context
    hidden = mx.sym.Concat(*[hidden, decoder_input], dim=1)
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)

    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label, weight=fc_weight, bias=fc_bias,
                               name='fc')
    sm = mx.sym.SoftmaxActivation(data=fc, name='decoder_softmax')

    output = [sm]
    for state in last_state:
        output.append(state.c)
        output.append(state.h)
    output.append(weighted_context)
    output.append(alpha)
    return mx.sym.Group(output)


def make_local_show_attend_and_tell_symbol(context_shape, seqlen, num_layer, num_hidden, num_label, vocab_size,
                                           num_embed, dropout=0.):
    encoder_input = mx.sym.Variable('encoder_input')
    decoder_input = mx.sym.Variable('decoder_input')
    decoder_output = mx.sym.Variable('decoder_output')
    mean_input = mx.sym.sum(data=encoder_input, axis=1, keepdims=False) / context_shape[0]
    context = encoder_input

    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    decoder_embed_weight = mx.sym.Variable('de_embed_weight')
    lo_attn_param = LocalAttnParam(lo_proj_h_weight=mx.sym.Variable('lo_proj_h_weight'))
    lo_concat_weight = mx.sym.Variable('lo_concat_weight')
    lstm_param_cells = []
    last_state = []
    for i in xrange(num_layer):
        lstm_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                          i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                          h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                          h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
        last_state.append(
            LSTMState(c=mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_c' % i),
                      h=mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_h' % i)))
    assert len(last_state) == num_layer

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

        alpha = make_local_attention_alpha(context, slice_input[seqidx], context_shape, lo_attn_param)
        weight_context = mx.sym.batch_dot(alpha, context)
        weight_context = mx.sym.Reshape(data=weight_context, shape=(-1, context_shape[1]))
        hidden = mx.sym.Concat(*[hidden, weight_context, slice_input[seqidx]], dim=1)
        # hidden = mx.sym.FullyConnected(data=hidden, num_hidden=num_hidden, weight=lo_concat_weight, no_bias=True)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label, weight=fc_weight, bias=fc_bias, name='fc')
    decoder_output = mx.sym.transpose(data=decoder_output)
    decoder_output = mx.sym.Reshape(data=decoder_output, shape=(-1,))

    sm = mx.sym.SoftmaxOutput(data=fc, label=decoder_output, name='decoder_softmax')
    return sm


def make_local_show_attend_and_tell_inference(context_shape, num_layer, num_hidden, num_label, vocab_size,
                                              num_embed, dropout=0., for_init=True):
    encoder_input = mx.sym.Variable('encoder_input')
    if for_init is True:
        mean_input = mx.sym.sum(data=encoder_input, axis=1, keepdims=False) / context_shape[0]
        output = []
        for i in xrange(num_layer):
            output.append(mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_c' % i))
            output.append(mx.sym.FullyConnected(data=mean_input, num_hidden=num_hidden, name='l%d_init_h' % i))
        return mx.sym.Group(output)

    decoder_input = mx.sym.Variable('decoder_input')
    context = encoder_input

    fc_weight = mx.sym.Variable('de_fc_weight')
    fc_bias = mx.sym.Variable('de_fc_bias')
    decoder_embed_weight = mx.sym.Variable('de_embed_weight')
    lo_attn_param = LocalAttnParam(lo_proj_h_weight=mx.sym.Variable('lo_proj_h_weight'))
    lo_concat_weight = mx.sym.Variable('lo_concat_weight')
    lstm_param_cells = []
    last_state = []
    for i in xrange(num_layer):
        lstm_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable('l%d_i2h_weight' % i),
                                          i2h_bias=mx.sym.Variable('l%d_i2h_bias' % i),
                                          h2h_weight=mx.sym.Variable('l%d_h2h_weight' % i),
                                          h2h_bias=mx.sym.Variable('l%d_h2h_bias' % i)))
        last_state.append(LSTMState(c=mx.sym.Variable('l%d_init_c' % i),
                                    h=mx.sym.Variable('l%d_init_h' % i)))
    assert len(last_state) == num_layer

    decoder_input = mx.sym.Embedding(data=decoder_input, input_dim=vocab_size, output_dim=num_embed,
                                     weight=decoder_embed_weight, name='decoder_embed')

    hidden = decoder_input
    for i in xrange(num_layer):
        if i == 0:
            dp_ratio = 0
        else:
            dp_ratio = dropout
        next_state = lstm(data=hidden, num_hidden=num_hidden, seqidx=0, layeridx=i,
                          param=lstm_param_cells[i], prev_state=last_state[i], dropout=dp_ratio)
        hidden = next_state.h
        last_state[i] = next_state
    alpha = make_local_attention_alpha(context, decoder_input, context_shape, lo_attn_param)
    weight_context = mx.sym.batch_dot(alpha, context)
    weight_context = mx.sym.Reshape(data=weight_context, shape=(-1, context_shape[1]))
    hidden = mx.sym.Concat(*[hidden, weight_context, decoder_input], dim=1)
    hidden = mx.sym.FullyConnected(data=hidden, num_hidden=num_hidden, weight=lo_concat_weight, no_bias=True)
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)

    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label, weight=fc_weight, bias=fc_bias,
                               name='fc')
    sm = mx.sym.SoftmaxActivation(data=fc, name='decoder_softmax')

    output = [sm]
    for state in last_state:
        output.append(state.c)
        output.append(state.h)
    output.append(alpha)
    return mx.sym.Group(output)
