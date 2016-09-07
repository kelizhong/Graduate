# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-8-17
# Time: 下午7:39
# Author: Zhu Danxiang
#

import os
import time
import logging
import json
import mxnet as mx
import numpy as np
from eval import score_helper


class CaptionTrainer(object):
    def __init__(self, ctx, symbol, options, ix2token, input_shape, metric, saver=None):
        self.ctx = ctx
        self.print_every = options.print_every
        self.ix2token = ix2token
        self.batch_size = options.batch_size
        self.input_shape = dict(input_shape)
        self.metric = metric
        self.saver = saver
        self.max_grad_norm = options.max_grad_norm
        optimizer = mx.optimizer.create(options.optim_method)
        optimizer.lr = options.learning_rate
        self.updater = mx.optimizer.get_updater(optimizer)
        if options.start_from_prefix is not None and options.start_from_epoch is not None:
            self.symbol, arg_dict, _ = mx.model.load_checkpoint(options.start_from_prefix, options.start_from_epoch)
            arg_name = self.symbol.list_arguments()
            for k, v in self.input_shape.iteritems():
                arg_dict[k] = mx.nd.zeros(v, ctx=self.ctx)
        else:
            self.initializer = mx.initializer.Uniform(scale=0.01)
            self.symbol = symbol
            arg_shape, _, _ = symbol.infer_shape(**self.input_shape)
            arg_name = symbol.list_arguments()
            arg_array = [mx.nd.zeros(x, ctx=ctx) for x in arg_shape]
            arg_dict = {k: v for k, v in zip(arg_name, arg_array)}
            for name in arg_name:
                if name in self.input_shape.keys():
                    continue
                self.initializer(name, arg_dict[name])
        grad_dict = {}
        for name in arg_name:
            if name in self.input_shape.keys():
                continue
            shape = arg_dict[name].shape
            grad_dict[name] = mx.nd.zeros(shape, ctx=ctx)
        self.executor = symbol.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='add')
        self.param_block = []
        for ix, name in enumerate(arg_name):
            if name in self.input_shape.keys():
                continue
            self.param_block.append((ix, arg_dict[name], grad_dict[name], name))

        self.trace = {}
        self.trace['options'] = {}
        self.trace['train'] = {}
        self.trace['validate'] = {}
        self.trace['train'][self.metric.__class__.__name__] = []
        self.trace['validate'][self.metric.__class__.__name__] = []
        for stype in score_helper.scorer_type:
            self.trace['validate'][stype] = []
        for k, v in vars(options).iteritems():
            self.trace['options'][k] = v

    def get_arg_dict(self):
        return {v[3]: v[1] for v in self.param_block}

    def get_grad_dict(self):
        return {v[3]: v[2] for v in self.param_block}

    def get_aux_dict(self):
        return self.executor.aux_dict

    def do_one_train_epoch(self, train_dataiter, epoch_num):
        assert self.executor is not None
        logging.info('Training Phase...')
        total_perf = 0.
        iter_num = 0
        tic = time.time()
        for batch_data in train_dataiter:
            self.do_one_iter(batch_data, is_train=True)
            pred = self.executor.outputs[0].asnumpy()
            label = batch_data.label[0].asnumpy()
            total_perf += self.metric.calculate(label, pred)
            iter_num += 1
            if iter_num % self.print_every == 0:
                self.trace['train'][self.metric.__class__.__name__].append(total_perf / self.batch_size)
                tac = time.time()
                speed = self.print_every * self.batch_size / (tac - tic)
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\t%s=%f',
                             epoch_num, iter_num, speed, self.metric.__class__.__name__, total_perf / self.batch_size)
                tic = time.time()
                total_perf = 0.

    def do_one_val_epoch(self, val_dataiter, epoch_num):
        logging.info('Validating Phase...')
        val_ref = []
        val_gold = []
        total_perf = 0.
        iter_num = 0
        tic = time.time()
        for batch_data in val_dataiter:
            self.do_one_iter(batch_data, is_train=False)
            pred = self.executor.outputs[0].asnumpy()
            label = batch_data.label[0].asnumpy()
            total_perf += self.metric.calculate(label, pred)
            pred = np.argmax(pred, axis=1)
            pred = np.reshape(pred, (-1, self.batch_size)).transpose((1, 0))
            val_ref.extend(
                [[self.convert_to_sentence(pred[i * 5 + j]) for j in xrange(5)] for i in xrange(len(pred) / 5)])
            val_gold.extend(
                [[self.convert_to_sentence(label[i * 5 + j]) for j in xrange(5)] for i in xrange(len(label) / 5)])
            iter_num += 1
        score = score_helper.compute_score(val_ref, val_gold)
        tac = time.time()
        self.trace['validate'][self.metric.__class__.__name__].append(total_perf / iter_num)
        logging.info('Epoch[%d]\tTime cost: %.2f sec\t%s=%f', epoch_num, (tac - tic), self.metric.__class__.__name__,
                     total_perf / iter_num)
        logging.info('Language Evaluate:')
        for m, v in score.iteritems():
            self.trace['validate'][m].append(v)
            logging.info('\t%s=%f', m, v)
        if self.saver is not None:
            self.saver.update(score['BLEU_1'], self.symbol, self.get_arg_dict(), self.get_aux_dict())

    def trace_dump(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        json.dump(self.trace, open(filename, 'w'))

    def convert_to_sentence(self, sequence):
        sent = []
        for id in sequence:
            token = self.ix2token[id]
            if token == '</s>':
                break
            else:
                sent.append(token)
        return ' '.join(sent)

    def do_one_iter(self, batch_data, is_train=True):
        for name, data in zip(batch_data.data_names, batch_data.data):
            data.copyto(self.executor.arg_dict[name])
        for name, label in zip(batch_data.label_names, batch_data.label):
            label.copyto(self.executor.arg_dict[name])
        self.executor.forward(is_train=is_train)
        if is_train is True:
            self.executor.backward()
            self.param_update()

    def param_update(self):
        norm = 0
        n_pred = self.executor.outputs[0].shape[0]
        for ix, weight, grad, name in self.param_block:
            grad /= n_pred
            l2_norm = mx.nd.norm(grad).asscalar()
            norm += l2_norm * l2_norm
        norm = np.sqrt(norm)
        for ix, weight, grad, name in self.param_block:
            if norm > self.max_grad_norm:
                grad *= (self.max_grad_norm / norm)
            self.updater(ix, grad, weight)
            grad[:] = 0.0


class CaptionInferencer(object):
    def __init__(self):
        super(CaptionInferencer, self).__init__()
