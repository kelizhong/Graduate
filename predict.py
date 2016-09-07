# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 上午9:39
# Author: Zhu Danxiang
#

import argparse
import logging
import cPickle
import h5py
import json
import mxnet as mx
import numpy as np
from core.inferences import ShowAndTellInference, StrictShowAttendAndTellInference


def load_testset(filename):
    with h5py.File(filename, 'r') as f:
        feats = f['feats'].value
        feat_shape = f.attrs['feat_shape']
        image_names = json.loads(f.attrs['image_names'])
    return feats, image_names, feat_shape


def save_output(filename, output):
    with open(filename, 'w') as f:
        for out in output:
            print >> f, '\t'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', help='test file', default='data/flickr8k-test.h5')
    parser.add_argument('--output_file', help='output file', default='data/flickr8k-test-predict.txt')
    parser.add_argument('--max_length', help='max generation length', default=20)
    parser.add_argument('--checkpoint_prefix', help='checkpoint prefix', default='checkpoint/nic')
    parser.add_argument('--checkpoint_epoch', help='checkpoint epoch', default=6)
    args = parser.parse_args()

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    config = cPickle.load(open(args.checkpoint_prefix + '-config.pkl', 'r'))
    ix2token = config['ix2token']
    token2ix = {k: v for v, k in ix2token.iteritems()}
    num_hidden = config['num_hidden']
    num_embed = config['num_embed']
    num_layer = config['num_layer']
    starter = token2ix['<s>']
    stopper = token2ix['</s>']
    vocab_size = len(token2ix)
    feats, image_names, feat_shape = load_testset(args.test_file)

    _, arg_params, _ = mx.model.load_checkpoint(args.checkpoint_prefix, args.checkpoint_epoch)
    # model = ShowAndTellInference(ctx=mx.gpu(), arg_params=arg_params, starter=starter, stopper=stopper,
    #                              max_length=args.max_length, feat_shape=feat_shape, vocab_size=vocab_size,
    #                              num_label=vocab_size, num_hidden=num_hidden, num_embed=num_embed, num_layer=num_layer,
    #                              dropout=0)
    model = StrictShowAttendAndTellInference(ctx=mx.gpu(), arg_params=arg_params, starter=starter, stopper=stopper,
                                             max_length=args.max_length, feat_shape=feat_shape, vocab_size=vocab_size,
                                             num_label=vocab_size, num_hidden=num_hidden, num_embed=num_embed,
                                             num_layer=num_layer, dropout=0)

    output = []
    for i in xrange(len(feats)):
        feat = feats[i]
        feat = np.reshape(feat, (1, feat_shape[0], feat_shape[1]))
        pred, attn = model.forward(feat)
        pred_str = ' '.join([ix2token[x] for x in pred])
        output.append((image_names[i], pred_str))

    save_output(args.output_file, output)
