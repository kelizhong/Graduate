# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-6
# Time: 下午9:19
# Author: Zhu Danxiang
#
import os
import argparse
import logging
import cPickle
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mxnet as mx
import numpy as np
from core.inferences import StrictShowAttendAndTellInference
from preprocess.feature_extractor import FeatureExtractor
from PIL import Image


def reshape(feat):
    feat_shape = feat[0].shape
    if len(feat_shape) == 1:
        feat = np.reshape(feat, (feat.shape[0], 1, feat.shape[1]))
    elif len(feat_shape) == 3:
        feat = np.reshape(feat, (feat.shape[0], feat.shape[1], -1)).transpose((0, 2, 1))
    return feat


def LoadImage(file_name, resize=256, crop=224):
    image = Image.open(file_name)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize
    left = (width - crop) / 2
    top = (height - crop) / 2
    image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
    data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
    data = data.astype('float32') / 255
    return data


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
                        default='checkpoint/flickr8k-att/2016-0906-1531/nic')
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
    layer = 'conv5_3'
    extractor = FeatureExtractor(args.vgg_deploy, args.vgg_model, args.vgg_mean, batch_size=1, layers=[layer],
                                 is_gpu=True)
    image_names = os.listdir(args.image_dir)
    feat_shape = reshape(extractor.extract_batch([args.image_dir + image_names[0]])[layer])[0].shape

    _, arg_params, _ = mx.model.load_checkpoint(args.checkpoint_prefix, args.checkpoint_epoch)

    model = StrictShowAttendAndTellInference(ctx=mx.gpu(), arg_params=arg_params, starter=starter, stopper=stopper,
                                             max_length=args.max_length, feat_shape=feat_shape, vocab_size=vocab_size,
                                             num_label=vocab_size, num_hidden=num_hidden, num_embed=num_embed,
                                             num_layer=num_layer, dropout=0)

    smooth = True

    for name in image_names:
        path = args.image_dir + name
        feat = extractor.extract_batch([path])
        feat = reshape(feat[layer])
        output, attn = model.forward(feat)
        output = [ix2token[x] for x in output]
        print '%s : %s' % (name, ' '.join(output))
        n_words = len(output) + 1
        w = np.round(np.sqrt(n_words))
        h = np.ceil(np.float32(n_words) / w)
        img = LoadImage(path)
        plt.subplot(w, h, 1)
        plt.imshow(img)
        plt.axis('off')

        for i in xrange(len(attn)):
            plt.subplot(w, h, i + 2)
            word = output[i]
            plt.text(0, 1, word, backgroundcolor='white', fontsize=13)
            plt.text(0, 1, word, color='black', fontsize=13)
            plt.imshow(img)
            alpha = attn[i].reshape(14, 14)
            if smooth:
                attn_img = skimage.transform.pyramid_expand(alpha, upscale=16, sigma=20)
            else:
                attn_img = skimage.transform.resize(alpha, [img.shape[0], img.shape[1]])
            plt.imshow(attn_img, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        plt.show()
