#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: 
"""
""
import sys
import numpy as np

def read_data(path, slot_indexes, slots_lengthes, delim=';', pad=0, type_dict=None):
    """read_data from disk, format as lego id file
    Args:
        path(string): path
    Returns:
        (Tuple):slots[0], slots[1], ..., slots[n_slots - 1]
    """
    n_slots = len(slot_indexes)
    slots = [[] for _ in xrange(n_slots)]
    if not type_dict:
        type_dict = {}

    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().split(delim)
            try:    
                i += 1
                if i % 10000 == 1:
                    print >> sys.stderr, 'read %d lines' % i
                raw = []
                for index in slot_indexes:
                    slot_value = items[index].split()
                    tp = type_dict.get(index, int)
                    raw.append([tp(x) for x in slot_value])

                for index in xrange(len(raw)):
                    slots[index].append(pad_and_trunc(raw[index],
                        slots_lengthes[index],
                        pad=pad,
                        sequence=slots_lengthes[index]>1))

            except Exception as e:
                print >> sys.stderr, '%s, Invalid data raw %s' % (e, line.strip())
                continue
    return slots


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield epoch, batch_num * 100.0 / num_batches_per_epoch, \
                    shuffled_data[start_index:end_index]


def pad_and_trunc(data, length, pad=0, sequence=False):
    """pad_and_trunc

    Args:
        data(list): .
        length(int): expect length
        pad(int): pad content
    Returns:
        type:
    """
    # Padding in left for sequence
    if pad < 0:
        return data
    if sequence: 
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)

    if len(data) > length:
        return data[:length]

    while len(data) < length:
        data.append(pad)
    return data
