# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-13
# Time: 下午2:00
# Author: Zhu Danxiang
#
import caffe
import skimage
import cv2
import numpy as np


class FeatureExtractor(object):
    def __init__(self, deploy, model, mean, cnn_type='vgg', batch_size=10, width=224, height=224, layers=['conv5_3'],
                 is_gpu=True):
        if cnn_type == 'vgg':
            self.cnn = VGG(deploy, model, mean, is_gpu)

        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.layers = layers

    def resize(self, image):
        assert len(image.shape) >= 2, 'image shape error.'
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image, 3, axis=2)
        resized_image = cv2.resize(image, (self.width, self.height))
        return resized_image

    def load(self, path):
        assert isinstance(path, str), 'path must be a string.'
        return skimage.img_as_float(skimage.io.imread(path)).astype(np.float32)

    def extract_all(self, image_paths):
        feats = {k: list() for k in self.layers}
        for ix in xrange(0, len(image_paths), self.batch_size):
            batch = min(self.batch_size, len(image_paths) - ix)
            path_batch = image_paths[ix: ix + batch]
            feat_batch = self.extract_batch(path_batch)
            map(lambda x: feats[x].extend(feat_batch[x]), feat_batch)
            if ix % 1000 == 0:
                print 'processed %d/%d' % (ix, len(image_paths))
        feats = {k: np.array(v) for k, v in feats.iteritems()}
        return feats

    def extract_batch(self, path_batch):
        image_batch = np.array(map(lambda x: self.load(x), path_batch))
        image_batch = np.array(map(lambda x: self.resize(x), image_batch))
        feat_batch = self.cnn.get_feature(image_batch, layers=self.layers)
        return feat_batch


class CNN(object):
    def __init__(self, deploy, model, mean, is_gpu=True):
        self.deploy = deploy
        self.model = model
        self.mean = mean
        self.is_gpu = is_gpu
        self.net, self.transformer = self.get_net()

    def get_net(self):
        if self.is_gpu is True:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        return net, transformer


class VGG(CNN):
    def __init__(self, deploy, model, mean, is_gpu=True):
        super(VGG, self).__init__(deploy, model, mean, is_gpu=is_gpu)

    def get_feature(self, image_batch, layers=['conv5_3']):
        caffe_in = np.zeros(np.array(image_batch.shape)[[0, 3, 1, 2]])
        for idx, in_ in enumerate(image_batch):
            caffe_in[idx] = self.transformer.preprocess('data', in_)
        out = self.net.forward_all(blobs=layers, **{'data': caffe_in})
        return {k: v for k, v in out.iteritems() if k in layers}
