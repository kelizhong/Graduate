# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 上午9:42
# Author: Zhu Danxiang
#

import mxnet as mx
import numpy as np
from encoder import make_sequence_encoder_inference, make_mean_encoder_inference
from decoder import make_sequence_decoder_inference
from symbols import make_strict_show_attend_and_tell_inference


class ShowAndTellInference(object):
    def __init__(self, ctx, arg_params, starter, stopper, max_length, feat_shape, vocab_size, num_label, num_hidden,
                 num_embed, num_layer, dropout=0.):
        self.starter = starter
        self.stopper = stopper
        self.max_length = max_length
        self.num_layer = num_layer
        encoder_input = mx.sym.Variable('encoder_input')
        decoder_input = mx.sym.Variable('decoder_input')
        self.encoder_inference = make_mean_encoder_inference(encoder_input=encoder_input, seqlen=1,
                                                             num_layer=num_layer, num_hidden=num_hidden,
                                                             dropout=dropout)
        self.decoder_inference = make_sequence_decoder_inference(decoder_input=decoder_input, num_layer=num_layer,
                                                                 num_hidden=num_hidden,
                                                                 num_label=num_label, dropout=dropout,
                                                                 vocab_size=vocab_size, num_embed=num_embed,
                                                                 with_embedding=True)
        batch_size = 1
        init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in xrange(num_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in xrange(num_layer)]
        self.encoder_shape = [('encoder_input', (batch_size, feat_shape[0], feat_shape[1]))]
        self.encoder_shape = dict(self.encoder_shape + init_c + init_h)
        self.decoder_shape = [('decoder_input', (batch_size,))]
        self.decoder_shape = dict(self.decoder_shape + init_c + init_h)

        self.encoder_executor = self.encoder_inference.simple_bind(ctx=ctx, **self.encoder_shape)
        self.decoder_executor = self.decoder_inference.simple_bind(ctx=ctx, **self.decoder_shape)
        for key in self.encoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.encoder_executor.arg_dict[key])
        for key in self.decoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.decoder_executor.arg_dict[key])

        state_name = []
        for i in xrange(num_layer):
            state_name.append('l%d_init_c' % i)
            state_name.append('l%d_init_h' % i)

        self.state_dict = dict(zip(state_name, self.decoder_executor.outputs[1:]))

    def forward(self, input_data):
        for key in self.state_dict.keys():
            self.encoder_executor.arg_dict[key][:] = 0.
            self.decoder_executor.arg_dict[key][:] = 0.
        if not self.check_shape_valid(input_data.shape):
            return
        input_data = mx.nd.array(input_data)
        input_data.copyto(self.encoder_executor.arg_dict['encoder_input'])
        self.encoder_executor.forward()

        for i in xrange(self.num_layer):
            self.encoder_executor.outputs[2 * i].copyto(self.decoder_executor.arg_dict['l%d_init_c' % i])
            self.encoder_executor.outputs[2 * i + 1].copyto(self.decoder_executor.arg_dict['l%d_init_h' % i])

        decoder_input = mx.nd.array([self.starter])
        output = []
        for i in xrange(self.max_length):
            decoder_input.copyto(self.decoder_executor.arg_dict['decoder_input'])
            self.decoder_executor.forward()
            prob = self.decoder_executor.outputs[0].asnumpy()
            token = np.argmax(prob, axis=1)
            if token == self.stopper:
                break
            output.append(token[0])
            decoder_input = mx.nd.array(token)
            for key in self.state_dict.keys():
                self.state_dict[key].copyto(self.decoder_executor.arg_dict[key])
        return output

    def check_shape_valid(self, input_shape):
        valid_shape = self.encoder_shape['encoder_input']
        if valid_shape == input_shape:
            return True
        else:
            print 'valid shape:', valid_shape, 'input shape:', input_shape
            return False


class StrictShowAttendAndTellInference(object):
    def __init__(self, ctx, arg_params, starter, stopper, max_length, feat_shape, vocab_size, num_label, num_hidden,
                 num_embed, num_layer, dropout=0.):
        self.starter = starter
        self.stopper = stopper
        self.max_length = max_length
        self.num_layer = num_layer
        self.encoder_inference = make_strict_show_attend_and_tell_inference(feat_shape, num_layer, num_hidden,
                                                                            num_label, vocab_size, num_embed, dropout,
                                                                            for_init=True)
        self.decoder_inference = make_strict_show_attend_and_tell_inference(feat_shape, num_layer, num_hidden,
                                                                            num_label, vocab_size, num_embed, dropout,
                                                                            for_init=False)
        batch_size = 1
        init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in xrange(num_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in xrange(num_layer)]
        init_state = init_c + init_h
        self.encoder_shape = [('encoder_input', (batch_size, feat_shape[0], feat_shape[1]))]
        self.decoder_shape = [('decoder_input', (batch_size,))] + self.encoder_shape + init_state
        self.encoder_shape = dict(self.encoder_shape)
        self.decoder_shape = dict(self.decoder_shape)

        self.encoder_executor = self.encoder_inference.simple_bind(ctx=ctx, **self.encoder_shape)
        self.decoder_executor = self.decoder_inference.simple_bind(ctx=ctx, **self.decoder_shape)

        for key in self.encoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.encoder_executor.arg_dict[key])
        for key in self.decoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.decoder_executor.arg_dict[key])

        state_name = []
        for i in xrange(num_layer):
            state_name.append('l%d_init_c' % i)
            state_name.append('l%d_init_h' % i)

        self.state_dict = dict(zip(state_name, self.decoder_executor.outputs[1:]))

    def forward(self, input_data):
        for key in self.state_dict.keys():
            self.decoder_executor.arg_dict[key][:] = 0.
        if not self.check_shape_valid(input_data.shape):
            return
        input_data = mx.nd.array(input_data)
        input_data.copyto(self.encoder_executor.arg_dict['encoder_input'])
        self.encoder_executor.forward()

        for i in xrange(self.num_layer):
            self.encoder_executor.outputs[2 * i].copyto(self.decoder_executor.arg_dict['l%d_init_c' % i])
            self.encoder_executor.outputs[2 * i + 1].copyto(self.decoder_executor.arg_dict['l%d_init_h' % i])
        decoder_input = mx.nd.array([self.starter])
        output = []
        attn = []
        for i in xrange(self.max_length):
            decoder_input.copyto(self.decoder_executor.arg_dict['decoder_input'])
            input_data.copyto(self.decoder_executor.arg_dict['encoder_input'])
            self.decoder_executor.forward()
            prob = self.decoder_executor.outputs[0].asnumpy()
            alpha = self.decoder_executor.outputs[-1].asnumpy()
            token = np.argmax(prob, axis=1)
            if token == self.stopper:
                break
            output.append(token[0])
            attn.append(alpha)
            decoder_input = mx.nd.array(token)
            for key in self.state_dict.keys():
                self.state_dict[key].copyto(self.decoder_executor.arg_dict[key])
        return output, attn

    def check_shape_valid(self, input_shape):
        valid_shape = self.encoder_shape['encoder_input']
        if valid_shape == input_shape:
            return True
        else:
            print 'valid shape:', valid_shape, 'input shape:', input_shape
            return False
