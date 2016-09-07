# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-8-30
# Time: 下午10:06
# Author: Zhu Danxiang
#

import json


def load_caption_info(filename):
    dataset = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for info in data['images']:
            if info.has_key('filepath'):
                image_name = info['filepath'] + '/' + info['filename']
            else:
                image_name = info['filename']
            sentences = [v['tokens'] for v in info['sentences']]
            split = info['split']
            dataset.append((image_name, sentences[:5], split))
    return dataset


def dump_caption_info(dataset, train_file, val_file, test_file):
    trainf = open(train_file, 'w')
    valf = open(val_file, 'w')
    testf = open(test_file, 'w')
    for info in dataset:
        for sent in info[1]:
            if info[2] == 'train':
                print >> trainf, '%s\t%s' % (info[0], ' '.join(sent))
            elif info[2] == 'val':
                print >> valf, '%s\t%s' % (info[0], ' '.join(sent))
            elif info[2] == 'test':
                print >> testf, '%s\t%s' % (info[0], ' '.join(sent))
    trainf.close()
    valf.close()
    testf.close()


if __name__ == '__main__':
    caption_file = '/home/pig/Data/CAPTION/dataset_coco.json'
    train_file = 'data/coco/train_coco_caption.txt'
    val_file = 'data/coco/val_coco_caption.txt'
    test_file = 'data/coco/test_coco_caption.txt'
    dataset = load_caption_info(caption_file)
    dump_caption_info(dataset, train_file, val_file, test_file)
