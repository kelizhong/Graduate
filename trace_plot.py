# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-9-4
# Time: 下午11:04
# Author: Zhu Danxiang
#

import sys
import argparse
import json
import matplotlib.pyplot as plt


def plot(perf):
    plt.plot(perf)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--log_file', help='log file', default='log/flickr8k-att/2016-0906-2222-log.json')
    options = parser.parse_args(sys.argv[1:])

    trace = json.load(open(options.log_file, 'r'))
    print max(trace['validate']['BLEU_1'])
    plot(trace['validate']['BLEU_1'])
