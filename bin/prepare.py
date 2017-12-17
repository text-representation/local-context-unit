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

def convert_multi_slots_to_single_slots(slots):
    """
    covert the data which text_data are saved as multi-slots, e.g()
    """
    if len(slots) <= 1:
        return slots
    else:
        return ' '.join(slots)


def build_vocab(save_dir, data, min_frq=2):
    """
    build vocab from data, which slot 1 is text data, slot 0 is label
    param:
        1: save_dir: the data dir where the vocabulary dict file saved
        2: data: input train data
        3: min_req : words that appare less than {min_req} times will be replaced as <unk>
    output:
        1: vocabulary_dict(key: word, value: idx)  
    """
    word_count = {}
    with open(data, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        for items in lines:
            text_data = convert_multi_slots_to_single_slots(items[1:])
            text_data = clean_str(text_data)
            words = set(text_data.split(' '))# word only counts once in a doucment
            for word in words:
                if word: #remove ' '
                    word_count[word] = word_count.get(word, 0) + 1

    # remove unfrequence words
    word_count_filterd = {k:v for k,v in word_count.items() if v >= min_frq}

    words_sorted_by_count = sorted(word_count_filterd.items(), key = lambda x:-x[1])

    vocab_dict = {"<pad>": "0", "<unk>" : '1'}
    with open(save_dir + '/unigram.id', 'w') as vocab_file:
        print >> vocab_file, "<pad>\t0"
        print >> vocab_file, "<unk>\t1"
        idx = 2
        for k, v in words_sorted_by_count:
            print >> vocab_file, k + '\t' + str(idx)
            vocab_dict[k] = str(idx)
            idx += 1
    return vocab_dict


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
                text_data = convert_multi_slots_to_single_slots(items[1:])
                text_data = clean_str(text_data)
                words = text_data.split(' ')
                word_ids = []
                for word in words:
                    word_ids.append(vocab.get(word, '1')) #1 == <unk>
                text_data_ids = ' '.join(word_ids)
                lable = str(int(items[0]) - 1) # add -1 bias to ensure min label value is 0
                print >> out_f, ';'.join([lable, text_data_ids])


def split_train_dev(data, dev_dir, rate):
    """
    split train data to dev_train and dev sets
    """
    with open(data, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        dev_idx = int(len(lines) * rate)

    train_file = dev_dir + '/dev_train.csv'
    dev_file = dev_dir + '/dev.csv'

    with open(dev_file, 'w') as f:
        f.write(''.join(lines[0:dev_idx]))

    with open(train_file, 'w') as f:
        f.write(''.join(lines[dev_idx:]))

    return train_file, dev_file

def process(train_file, test_file, frequence_cut, save_dir, dev_file=''):

    """
    build vocabulary dict from train_file
    tokenlize all datas
    """
    if not os.path.exists(v):
        print >> sys.stderr, 'vocab not exist, build from data'
        vocab = build_vocab(save_dir, train_file, frequence_cut)
    else:
        vocab = load_vocab(v)
    for _file in [train_file, test_file]:
        token_lize(vocab, _file)
    if dev_file:
        token_lize(vocab, dev_file)


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
    frequence_cut = 2 #remove words that only apear in one document
    if not options: _usage()
    for key, val in options:
        val = val.strip()
        if key == '--vocab':
            v = val
        elif key == '--data_dir':
            data_dir = val
        elif key == '--min_frq_cut':
            frequence_cut = int(val)
        elif key == '--dev_rate':
            dev_rate = float(val)
        elif key == '-h' or key == '--help':
            _usage()
        else: _usage()

    dev_dir = data_dir.strip('/') + '_for_dev'
    os.system('mkdir -p %s' % dev_dir)
    os.system('cp %s/test.csv %s/dev_test.csv' % (data_dir, dev_dir))

    dev_train_file, dev_file = split_train_dev(data_dir + '/train.csv', dev_dir, dev_rate)
    dev_test_file = dev_dir + '/dev_test.csv'
    
    test_file = data_dir + '/test.csv'
    train_file = data_dir + '/train.csv'

    #generate data for parameter tuning, default using 10% training data for dev
    process(dev_train_file, dev_test_file, frequence_cut, dev_dir, dev_file)

    #use the best parameter and 100% training data to get the model's performance
    process(train_file, test_file, frequence_cut, data_dir)
