#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author	:

Date	:

Brief	: 
"""

import sys
import os
import tensorflow as tf

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
lib_dir = os.path.abspath(os.path.join(base_dir, 'lib'))
sys.path.append(lib_dir)

from tedll import layers

class RegionEmbeddingClassify(object):
    """RegionEmbeddingClassify"""
    def __init__(self, config):
        self._config = config
        
        fn_dict = {
                    'reduce_max': tf.reduce_max, \
                    'reduce_sum': tf.reduce_sum, \
                    'concat': tf.reshape}

        win_merge_fn = fn_dict.get(config.win_merge_fn, tf.reduce_max)

        # Layers
        assert(config.mode in ['WC', 'CW'])

        if config.mode == 'WC':
            L = layers.WordContextRegionEmbeddingLayer
        if config.mode == 'CW':
            L = layers.ContextWordRegionEmbeddingLayer

        self._region_emb_layer = L(config.vocab_size,
                config.emb_size,
                config.win_size,
                win_merge_fn=win_merge_fn)

        self._vsum_layer = layers.WeightedVSumLayer()

        self._fc_layer = layers.FCLayer(config.hidden_depth, config.n_classes)

        # Inputs
        self.sequence = tf.placeholder(tf.int32, \
                            [None, config.max_sequence_length], \
                            name='sequence')

        self.label = tf.placeholder(tf.int64, \
                    [None, 1], \
                    name='label')

        # Fetch OPs
        self.logits_op = self.logits()
        self.loss_op = self.loss()

    def logits(self):
        """logits"""
        
        region_emb = self._region_emb_layer(self.sequence)
       
        # Mask padding elements (id > 0) 
        win_radius = self._config.win_size / 2
        #trimed_seq = self.sequence[:, win_radius : self.sequence.get_shape()[1] - win_radius]
        trimed_seq = self.sequence[..., win_radius: self.sequence.get_shape()[1] - win_radius]
        def mask(x):
            """mask
            """
            return tf.cast(tf.greater(tf.cast(x, tf.int32), tf.constant(0)), tf.float32)
        weight = tf.map_fn(mask, trimed_seq, dtype=tf.float32, back_prop=False)
        weight = tf.expand_dims(weight, -1)
        # End mask

        h = self._vsum_layer((region_emb, weight))
        h = self._fc_layer(h)
        return h

    def loss(self):
        """loss"""
        logits = self.logits_op
        label = tf.one_hot(self.label, depth=self._config.n_classes)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
        return loss


def main():
    """main"""
    pass

if '__main__' == __name__:
    main()

