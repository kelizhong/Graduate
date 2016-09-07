# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-9-4
# Time: 上午10:31
# Author: Zhu Danxiang
#

import sys
import argparse
import time
import logging
import cPickle
import mxnet as mx
import numpy as np
from datamanager import CaptionIter
from core.symbols import make_show_and_tell_symbol, make_show_attend_and_tell_symbol, \
    make_strict_show_attend_and_tell_symbol
from model_helper import CaptionTrainer
from util.metric import Perplexity, NegativeLogLikehood, Accuracy
from util.saver import BestScoreSaver


def load_vocab(filename):
    token2ix = dict()
    ix2token = dict()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            ix, token = line.split()
            token2ix[token] = int(ix)
            ix2token[int(ix)] = token
    return token2ix, ix2token


if __name__ == '__main__':

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--base_data_path', help='base path which contains all data', default='data')
    parser.add_argument('--data_prefix', help='which dataset you want to use? flickr8k, flickr30k, coco',
                        default='flickr8k-att')
    parser.add_argument('--checkpoint_base_path',
                        help='checkpoint base path, a time stamp will be forced to insert in the end',
                        default='checkpoint')
    parser.add_argument('--checkpoint_prefix', help='checkpoint prefix, using to save model', default='nic')
    parser.add_argument('--start_from_prefix', help='start from a checkpoint prefix',
                        default=None)
    parser.add_argument('--start_from_epoch', help='start from a checkpoint epoch', default=None)
    parser.add_argument('--log_base_path', help='log file base path', default='log')
    parser.add_argument('--batch_size', help='batch size', default=150)
    parser.add_argument('--num_layer', help='number of rnn stack layer', default=1)
    parser.add_argument('--num_hidden', help='number of hidden unit in rnn', default=512)
    parser.add_argument('--num_embed', help='number of embedding unit', default=512)
    parser.add_argument('--num_epoch', help='number of epoch', default=500)
    parser.add_argument('--dropout', help='dropout ratio', default=0.5)
    parser.add_argument('--learning_rate', help='learning rate', default=0.0001)
    parser.add_argument('--max_grad_norm', help='max gradient norm', default=5)
    parser.add_argument('--optim_method', help='optimization method, default is adam', default='adam')
    parser.add_argument('--metric', help='performance metric: perplexity, nll, accuracy', default='perplexity')
    parser.add_argument('--print_every', help='how many batch iteration to print one performance', default=50)
    parser.add_argument('--with_image_att',
                        help='whether with image attention or not. No attention means that you should init rnn unit',
                        default=True)
    parser.add_argument('--verbose_print', help='print the generated sentences when evaluating performance',
                        default=False)
    parser.add_argument('--gpuid', help='gpu device id, -1 is cpu', default=0)
    parser.add_argument('--seed', help='random seed', default=123)

    options = parser.parse_args(sys.argv[1:])

    if isinstance(options.seed, int):
        mx.random.seed(options.seed)
    else:
        mx.random.seed(123)

    if options.gpuid == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(options.gpuid)

    data_prefix = options.base_data_path + '/' + options.data_prefix + '/' + options.data_prefix
    train_file = data_prefix + '-train.h5'
    val_file = data_prefix + '-test.h5'
    vocab_file = data_prefix + '-dict.txt'
    time_stamp = time.strftime('%Y-%m%d-%H%M', time.localtime())
    log_file = options.log_base_path + '/' + options.data_prefix + '/' + time_stamp + '-log.json'
    checkpoint_prefix = options.checkpoint_base_path + '/' + options.data_prefix + '/' + time_stamp + '/' + options.checkpoint_prefix
    token2ix, ix2token = load_vocab(vocab_file)
    vocab_size = len(token2ix)
    if options.with_image_att is False:
        init_c = [('l%d_init_c' % l, (options.batch_size, options.num_hidden)) for l in xrange(options.num_layer)]
        init_h = [('l%d_init_h' % l, (options.batch_size, options.num_hidden)) for l in xrange(options.num_layer)]
        init_state = init_c + init_h
    else:
        init_state = []

    train_dataiter = CaptionIter(train_file, init_state, options.batch_size)
    val_dataiter = CaptionIter(val_file, init_state, options.batch_size)

    if options.with_image_att is False:
        sym = make_show_and_tell_symbol(train_dataiter.max_length, options.num_layer, options.num_hidden, vocab_size,
                                        vocab_size, options.num_embed, options.dropout)
    else:
        sym = make_strict_show_attend_and_tell_symbol(train_dataiter.feat_shape, train_dataiter.max_length,
                                                      options.num_layer, options.num_hidden, vocab_size,
                                                      vocab_size, options.num_embed, options.dropout)

    config = dict()
    config['num_layer'] = options.num_layer
    config['num_hidden'] = options.num_hidden
    config['num_embed'] = options.num_embed
    config['ix2token'] = ix2token
    if options.metric == 'perplexity':
        metric = Perplexity(verbose=options.verbose_print)
    elif options.metric == 'nll':
        metric = NegativeLogLikehood(verbose=options.verbose_print)
    elif options.metric == 'accuracy':
        metric = Accuracy(verbose=options.verbose_print)
    else:
        raise ValueError('metric error, metric must in perplexity, nll, accuracy')
    input_shape = dict(train_dataiter.provide_data + train_dataiter.provide_label)
    saver = BestScoreSaver(checkpoint_prefix, options.num_epoch, config)
    trainer = CaptionTrainer(ctx, sym, options, ix2token, input_shape, metric, saver)

    for i in xrange(options.num_epoch):
        trainer.do_one_train_epoch(train_dataiter, i)
        trainer.do_one_val_epoch(val_dataiter, i)
        trainer.trace_dump(log_file)
