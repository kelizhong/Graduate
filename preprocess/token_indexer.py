# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-13
# Time: 下午2:37
# Author: Zhu Danxiang
#
import nltk
from collections import defaultdict


class TokenIndexer(object):
    def __init__(self, symbols=["*blank*", "<unk>", "<s>", "</s>"], lowercase=True, stemming=False):
        self.symbols = symbols
        self.lowercase = lowercase
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.token_counter = defaultdict(int)
        self.vocab = None
        self.token2ix = None
        self.ix2token = None
        self.is_updated = None

    def append_tokens(self, tokens):
        for token in tokens:
            token = self.normalize(token)
            self.token_counter[token] += 1
        self.is_updated = False

    def append_sentences(self, sentences):
        for sent in sentences:
            if isinstance(sent, str):
                sent = sent.strip().split()
            elif isinstance(sent, list):
                sent = sent
            else:
                raise ValueError
            self.append_tokens(sent)
        self.is_updated = False

    def extend(self, indexer):
        tc = indexer.token_counter
        for k, v in tc.iteritems():
            k = self.normalize(k)
            self.token_counter[k] += v
        self.is_updated = False

    def flush_vocab(self):
        self.vocab = self.symbols + self.token_counter.keys()
        self.is_updated = True

    def get_vocab(self):
        assert self.vocab is not None, 'you should append some text first.'
        if self.is_updated is False:
            self.flush_vocab()
        return self.vocab

    def token_to_ix(self):
        if self.is_updated is False:
            self.flush_vocab()
        token2ix = dict()
        ix = 0
        for token in self.vocab:
            token2ix[token] = ix
            ix += 1
        return token2ix

    def ix_to_token(self):
        if self.is_updated is False:
            self.flush_vocab()
        ix2token = dict()
        ix = 0
        for token in self.vocab:
            ix2token[ix] = token
            ix += 1
        return ix2token

    def filter_and_prune(self, min_freq=1, max_vocab=10000):
        tf = sorted(self.token_counter.iteritems(), key=lambda x: x[1], reverse=True)
        self.token_counter = defaultdict()
        for i, token in enumerate(tf):
            if i < max_vocab and token[1] >= min_freq:
                self.token_counter[token[0]] = token[1]
        self.is_updated = False

    def convert_token(self, token):
        if self.is_updated is False:
            self.flush_vocab()
        if self.token2ix is None or len(self.vocab) != len(self.token2ix):
            self.token2ix = self.token_to_ix()
        return self.token2ix[token] if token in self.token2ix else self.token2ix[self.UNK]

    def convert_sentence(self, sentence):
        assert isinstance(sentence, list), 'sentence must be a list.'
        if self.is_updated is False:
            self.flush_vocab()
        return [self.convert_token(token) for token in sentence]

    def pad_sentence(self, sentence, length, symbol):
        assert isinstance(sentence, list), 'sentence must be a list.'
        if len(sentence) > length:
            return sentence
        else:
            return sentence + [symbol] * (length - len(sentence))

    def convert_fixed_length_sentence(self, sentence, length):
        assert isinstance(sentence, list), 'sentence must be a list.'
        if len(sentence) > length:
            sentence = sentence[:length]
        sentence = self.pad_sentence(sentence, length, self.PAD)
        sentence = self.convert_sentence(sentence)
        return sentence

    def clean(self, sentence):
        assert isinstance(sentence, str), 'clean sentence must be string.'
        sentence = sentence.replace(self.PAD, '')
        sentence = sentence.replace(self.BOS, '')
        sentence = sentence.replace(self.EOS, '')
        sentence = sentence.strip()
        return sentence

    def normalize(self, token):
        if self.lowercase is True:
            token = token.lower()
        if self.stemming is True:
            token = self.stemmer.stem(token)
        return token

    def write_counter(self, filename):
        with open(filename, 'w') as f:
            counter = sorted(self.token_counter.iteritems(), key=lambda x: x[1], reverse=True)
            for k, v in counter:
                print >> f, k, v

    def write_ix2token(self, filename):
        if self.is_updated is False:
            self.flush_vocab()
        if self.ix2token is None or len(self.vocab) != len(self.ix2token):
            self.ix2token = self.ix_to_token()
        with open(filename, 'w') as f:
            for k, v in self.ix2token.iteritems():
                print >> f, k, v

    def write_token2ix(self, filename):
        if self.is_updated is False:
            self.flush_vocab()
        if self.token2ix is None or len(self.vocab) != len(self.token2ix):
            self.token2ix = self.token_to_ix()
        with open(filename, 'w') as f:
            for k, v in self.token2ix.iteritems():
                print >> f, k, v

    def load_counter(self, filename):
        self.token_counter = defaultdict()
        with open(filename, 'r') as f:
            for line in f:
                k, v = line.strip().split()
                self.token_counter[k] = int(v)
        self.is_updated = False
