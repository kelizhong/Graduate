# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-8-31
# Time: 下午4:00
# Author: Zhu Danxiang
#

import json
import argparse
import sys
import os
import h5py
import numpy as np
from preprocess.feature_extractor import FeatureExtractor
from preprocess.token_indexer import TokenIndexer


class CaptionInfo(object):
    def __init__(self, image_name, sentences, split):
        self.image_name = image_name
        self.sentences = sentences[:5]
        self.split = split


def load_caption_info(filename):
    capset = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for info in data['images']:
            if info.has_key('filepath'):
                image_name = info['filepath'] + '/' + info['filename']
            else:
                image_name = info['filename']
            sentences = [v['tokens'] for v in info['sentences']]
            split = info['split']
            if len(sentences) < 5:
                continue
            capset.append(CaptionInfo(image_name, sentences, split))
    return capset


def make_indexer(capset):
    """
    构建索引器indexer，用于保存词表和一些处理caption的操作
    """
    indexer = TokenIndexer()
    for capinfo in capset:
        for sent in capinfo.sentences:
            indexer.append_tokens(sent)
    return indexer


def build_dataset(capset, extractor, indexer, image_base_dir, filename, batch_size, max_length, split=['train'],
                  layer='conv5_3'):
    num_image = 0
    num_cap = 0
    for capinfo in capset:
        if capinfo.split in split:
            num_image += 1
            num_cap += len(capinfo.sentences)
    assert num_cap == 5 * num_image
    if layer == 'conv5_3':
        feat_shape = (196, 512)
    elif layer == 'fc7':
        feat_shape = (1, 4096)
    f = h5py.File(filename, 'w')
    feats = f.create_dataset('feats', (num_image, feat_shape[0], feat_shape[1]), dtype='float32')
    caps = f.create_dataset('caps', (num_cap, max_length), dtype='uint32')
    start_ix = f.create_dataset('start_ix', (num_image,), dtype='uint32')
    end_ix = f.create_dataset('end_ix', (num_image,), dtype='uint32')
    caplen = f.create_dataset('caplen', (num_cap,), dtype='uint8')
    f.attrs['num_image'] = num_image
    f.attrs['num_caption'] = num_cap
    f.attrs['max_length'] = max_length
    f.attrs['feat_shape'] = feat_shape
    image_idx = 0
    cap_idx = 0
    path_batch = []
    cap_batch = []
    for capinfo in capset:
        if capinfo.split in split:
            path_batch.append(str(image_base_dir + capinfo.image_name))
            cap_batch.extend(capinfo.sentences)
        if len(path_batch) == batch_size:
            caption = np.array(
                [indexer.convert_fixed_length_sentence([indexer.BOS] + cap + [indexer.EOS], max_length) for cap in
                 cap_batch])
            _caplen = np.array([min(len(cap) + 2, max_length) for cap in cap_batch])
            _start_ix = np.array([cap_idx + i * 5 for i in xrange(batch_size)])
            _end_ix = np.array([cap_idx + (i + 1) * 5 for i in xrange(batch_size)])
            feature = extractor.extract_batch(path_batch)[layer]
            if layer == 'fc7':
                feature = np.reshape(feature, (batch_size, feat_shape[0], feat_shape[1]))
            elif layer == 'conv5_3':
                feature = np.reshape(feature, (batch_size, feat_shape[1], feat_shape[0])).transpose((0, 2, 1))
            feats[image_idx: image_idx + len(feature)] = feature
            caps[cap_idx: cap_idx + len(caption)] = caption
            start_ix[image_idx: image_idx + len(feature)] = _start_ix
            end_ix[image_idx: image_idx + len(feature)] = _end_ix
            caplen[cap_idx: cap_idx + len(caption)] = _caplen
            image_idx += len(feature)
            cap_idx += len(caption)
            path_batch = []
            cap_batch = []
            if image_idx % 1000 == 0:
                print 'processd %d/%d images (%.2f%% done)' % (image_idx, num_image, image_idx * 100.0 / num_image)
    f.close()


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--caption_info_file', help='caption info file which contains image path and its caption',
                        default='/home/pig/Data/CAPTION/dataset_flickr8k.json')
    parser.add_argument('--vgg_model_file', help='vgg model path',
                        default='/home/pig/Data/VGG/VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('--vgg_deploy_file', help='vgg deploy file path',
                        default='/home/pig/Data/VGG/VGG_ILSVRC_16_layers_deploy.prototxt')
    parser.add_argument('--image_mean_file', help='image mean file',
                        default='/usr/local/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    parser.add_argument('--image_base_dir', help='image base directory',
                        default='/home/pig/Data/FLICKR8K/flickr8k/Flicker8k_Dataset/')
    parser.add_argument('--feature_layer', help='which layer of feature you want to extract', default='fc7')
    parser.add_argument('--output_prefix', help='output prefix', default='flickr8k')
    parser.add_argument('--max_length', help='max length for each sentence', default=16)

    options = parser.parse_args(arguments)

    if not os.path.exists('data/' + options.output_prefix):
        os.makedirs('data/' + options.output_prefix)
    base_path = os.getcwd()
    train_save_path = os.path.join(base_path, 'data/' + options.output_prefix, options.output_prefix + '-train.h5')
    test_save_path = os.path.join(base_path, 'data/' + options.output_prefix, options.output_prefix + '-test.h5')
    indexer_save_path = os.path.join(base_path, 'data/' + options.output_prefix, options.output_prefix + '-dict.txt')

    capset = load_caption_info(options.caption_info_file)
    print 'Finish loading caption info...'
    indexer = make_indexer(capset)
    print 'Finish making indexer...'
    indexer.write_ix2token(indexer_save_path)

    extractor = FeatureExtractor(options.vgg_deploy_file, options.vgg_model_file, options.image_mean_file,
                                 batch_size=200, layers=[options.feature_layer], is_gpu=True)

    build_dataset(capset, extractor, indexer, options.image_base_dir, train_save_path, 200, options.max_length,
                  ['train', 'val', 'restval'], options.feature_layer)
    build_dataset(capset, extractor, indexer, options.image_base_dir, test_save_path, 200, options.max_length, ['test'],
                  options.feature_layer)


if __name__ == '__main__':
    main(sys.argv[1:])
