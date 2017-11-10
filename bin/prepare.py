"""
description:
author: 
"""

import random
import sys
import getopt
import os
import re
import csv

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(data_dir, data, min_frq=2):
    """
    build vocab from data, which slot 1 is text data, slot 0 is label
    """
    dic = {}
    with open(data, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        for items in lines:
            used = set()#word only count once in a doucment 
            items[1] = clean_str(items[1])
            subs = items[1].split(' ')
            for ite in subs:
                if ite in used:
                    continue
                used.add(ite)
                if ite:
                    dic[ite] = dic.get(ite, 0) + 1
    res = {}
    unk = 0
    for k in dic:
        if dic[k] >= min_frq:
            res[k] = dic[k]
        else:
            unk += 1
    cnts = sorted(res.items(), key = lambda x:-x[1])
    v_id = {}
    with open(data_dir + '/unigram.id', 'w') as id_f:
        print >> id_f, "<unk>\t1"
        print >> id_f, "<pad>\t0"
        v_id['<unk>'] = '1'
        v_id['<pad>'] = '0'
        i = 2
        for k, v in cnts:
            print >> id_f, k + '\t' + str(i)
            v_id[k] = str(i)
            i += 1
    return v_id


def load_vocab(v):
    """
    load vocabulary whcih is builded previously
    """
    res = {}
    for line in open(v):
        items = line.strip().split('\t')
        res[items[0]] = items[1]
    return res


def token_lize(vocab, data):
    """
    convert word to corresponding vocabulary id
    """
    with open(data + '.id', "w") as out_f:
        with open(data, 'rb') as csvfile:
            lines = csv.reader(csvfile)
            for items in lines:
                items[1] = clean_str(items[1])
                subs = items[1].split(' ')
                s = []
                for i in xrange(len(subs)):
                    s.append(vocab.get(subs[i], '1')) #1 == <unk>
                items[1] = ' '.join(s)
                items[0] = str(int(items[0]) - 1) # add -1 bias to ensure min label value is 0
                print >> out_f, ';'.join(items)


def split_train_dev(data, rate):
    """
    split data to train and dev sets
    """
    with open(data, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        dev_idx = int(len(lines) * rate)

    train_file = data + '.train'
    dev_file = data + '.dev'

    with open(dev_file, 'w') as f:
        f.write(''.join(lines[0:dev_idx]))

    with open(train_file, 'w') as f:
        f.write(''.join(lines[dev_idx:]))

    return train_file, dev_file


if __name__ == '__main__':

    def _usage():
        print >> sys.stderr, "Usage: python %s --data_dir=data_path " % (sys.argv[0])
        exit(-1)
    try:
        options, paths = getopt.getopt(sys.argv[1:], 'h',\
                ['help', "data_dir=", "min_frq_cut=", "dev_rate="])
    except:_usage()

    v = ''
    data_dir = ''
    dev_rate = 0.1
    cut = 2 #remove words that only apear in one document
    if not options: _usage()
    for key, val in options:
        val = val.strip()
        if key == '--vocab':
            v = val
        elif key == '--data_dir':
            data_dir = val
        elif key == '--min_frq_cut':
            cut = int(val)
        elif key == '--dev_rate':
            dev_rate = float(val)
        elif key == '-h' or key == '--help':
            _usage()
        else: _usage()

    train_file, dev_file = split_train_dev(data_dir + '/train.csv', dev_rate)
    test_file = data_dir + '/test.csv'

    if not os.path.exists(v):
        print >> sys.stderr, 'vocab not exist, build from data'
        vocab = build_vocab(data_dir, train_file, cut)
    else:
        vocab = load_vocab(v)
    for _file in [train_file, dev_file, test_file]:
        token_lize(vocab, _file)
