# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 上午11:20
# Author: Zhu Danxiang
#

import os
import argparse
import logging
import cPickle
import skimage.io
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from core.inferences import ShowAndTellInference
from preprocess.feature_extractor import FeatureExtractor


def reshape(feat):
    feat_shape = feat[0].shape
    if len(feat_shape) == 1:
        feat = np.reshape(feat, (feat.shape[0], 1, feat.shape[1]))
    elif len(feat_shape) == 3:
        feat = np.reshape(feat, (feat.shape[0], feat.shape[1], -1)).transpose((0, 2, 1))
    return feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='test file', default='/home/pig/Data/FLICKR8K/flickr8k/test/')
    parser.add_argument('--vgg_model', help='vgg model file path',
                        default='/home/pig/Data/VGG/VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('--vgg_deploy', help='vgg deploy file path',
                        default='/home/pig/Data/VGG/VGG_ILSVRC_16_layers_deploy.prototxt')
    parser.add_argument('--vgg_mean', help='mean file',
                        default='/usr/local/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    parser.add_argument('--max_length', help='max generation length', default=20)
    parser.add_argument('--checkpoint_prefix', help='checkpoint prefix',
                        default='checkpoint/flickr8k/2016-0905-1638/nic')
    parser.add_argument('--checkpoint_epoch', help='checkpoint epoch', default=500)
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
    layer = 'fc7'
    extractor = FeatureExtractor(args.vgg_deploy, args.vgg_model, args.vgg_mean, batch_size=1, layers=[layer],
                                 is_gpu=True)
    image_names = os.listdir(args.image_dir)
    feat_shape = reshape(extractor.extract_batch([args.image_dir + image_names[0]])[layer])[0].shape
    _, arg_params, _ = mx.model.load_checkpoint(args.checkpoint_prefix, args.checkpoint_epoch)
    model = ShowAndTellInference(ctx=mx.gpu(), arg_params=arg_params, starter=starter, stopper=stopper,
                                 max_length=args.max_length, feat_shape=feat_shape, vocab_size=vocab_size,
                                 num_label=vocab_size, num_hidden=num_hidden, num_embed=num_embed, num_layer=num_layer,
                                 dropout=0)

    for name in image_names:
        path = args.image_dir + name
        feat = extractor.extract_batch([path])
        feat = reshape(feat[layer])
        output = model.forward(feat)
        print '%s : %s' % (name, ' '.join([ix2token[x] for x in output]))
        img = skimage.io.imread(path)
        skimage.io.imshow(img)
        plt.show()
