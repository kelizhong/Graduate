# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-7-1
# Time: 下午2:33
# Author: Zhu Danxiang
#

import argparse
from eval import score_helper


def load_im2caption(filename):
    im2caption = dict()
    with open(filename, 'r') as f:
        for line in f:
            image_name, caption = line.strip().split('\t')
            if image_name in im2caption:
                im2caption[image_name].append(caption)
            else:
                im2caption[image_name] = [caption]
    return im2caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', help='test file', default='data/test_flickr8k_caption.txt')
    parser.add_argument('--predict_file', help='output file', default='data/flickr8k-test-predict.txt')
    args = parser.parse_args()

    gold_dict = load_im2caption(args.gold_file)
    pred_dict = load_im2caption(args.predict_file)

    commen_key = set(gold_dict.keys()) & set(pred_dict.keys())
    gold = []
    pred = []
    for key in commen_key:
        gold.append(gold_dict[key])
        pred.append(pred_dict[key])

    score = score_helper.compute_score(pred, gold)
    for m, v in score:
        print m, v
