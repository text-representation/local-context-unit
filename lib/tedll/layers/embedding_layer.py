#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author	:

Date	:

Brief	: 
"""

import tensorflow as tf
import numpy as np
from . import Layer
from . import RegionAlignmentLayer

class EmbeddingLayer(Layer):
    """EmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, name="embedding", 
            initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._W = self.get_variable(name + '_W', shape=[vocab_size, emb_size],
                initializer=initializer)

    def _forward(self, seq):
        return tf.nn.embedding_lookup(self._W, seq)


class WindowPoolEmbeddingLayer(EmbeddingLayer):
    """WindowPoolEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size, \
            region_merge_fn=None, \
            name="win_pool_embedding", \
            initializer=None, \
            **kwargs):
        
        Layer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, emb_size],
                initializer=initializer)
        super(WindowPoolEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(WindowPoolEmbeddingLayer, self)._forward(region_aligned_seq)

        return self._region_merge_fn(region_aligned_emb, axis=2)


class ScalarRegionEmbeddingLayer(EmbeddingLayer):
    """WordContextRegionEmbeddingLayer(Scalar)"""
    def __init__(self, vocab_size, emb_size, region_size, \
            region_merge_fn=None, \
            name="scalar_region_embedding", \
            initializer=None, \
            **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, 1],
                initializer=initializer)
        super(ScalarRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(ScalarRegionEmbeddingLayer, self)._forward(region_aligned_seq)

        region_radius = self._region_size / 2
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = tf.nn.embedding_lookup(self._K, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return self._region_merge_fn(projected_emb, axis=2)


class MultiRegionEmbeddingLayer(EmbeddingLayer):
    """"WordContextRegionEmbeddingLayer(Multi-region)"""
    def __init__(self, vocab_size, emb_size, region_sizes, \
            region_merge_fn=None, \
            name="multi_region_embedding", \
            initializer=None, \
            **kwargs):
        
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_sizes = region_sizes[:]
        self._region_sizes.sort()
        self._region_merge_fn = region_merge_fn
        region_num = len(region_sizes)

        self._K = [None] * region_num
        self._K[-1] = tf.get_variable(name + '_K_%d' % (region_num - 1), \
                    shape=[vocab_size, self._region_sizes[-1], emb_size], \
                    initializer=initializer)

        for i in range(region_num - 1):
            st = self._region_sizes[-1]/2 - self._region_sizes[i]/2
            ed = st + self._region_sizes[i]
            self._K[i] = self._K[-1][:, st:ed, :]

        super(MultiRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        """_forward
        """

        multi_region_emb = [] 

        for i, region_kernel in enumerate(self._K):
            region_radius = self._region_sizes[i] / 2
            region_aligned_seq = RegionAlignmentLayer(self._region_sizes[i], name="RegionAlig_%d" % (i))(seq)
            region_aligned_emb = super(MultiRegionEmbeddingLayer, self)._forward(region_aligned_seq)
             
            trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
            context_unit = tf.nn.embedding_lookup(region_kernel, trimed_seq)

            projected_emb = region_aligned_emb * context_unit
            region_emb =  self._region_merge_fn(projected_emb, axis=2)
            multi_region_emb.append(region_emb)
        
        return multi_region_emb


class WordContextRegionEmbeddingLayer(EmbeddingLayer):
    """WordContextRegionEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size, \
            region_merge_fn=None, \
            name="word_context_region_embedding", \
            initializer=None, \
            **kwargs):
        Layer.__init__(self, name, **kwargs) 
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable(name + '_K', shape=[vocab_size, region_size, emb_size],
                initializer=initializer)
        super(WordContextRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(WordContextRegionEmbeddingLayer, self)._forward(region_aligned_seq)

        region_radius = self._region_size / 2
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = tf.nn.embedding_lookup(self._K, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return self._region_merge_fn(projected_emb, axis=2)


class ContextWordRegionEmbeddingLayer(EmbeddingLayer):
    """ContextWordRegionEmbeddingLayer"""
    def __init__(self, vocab_size, emb_size, region_size, 
            region_merge_fn=None,
            name="embedding",
            initializer=None, **kwargs):
        super(ContextWordRegionEmbeddingLayer, self).__init__(vocab_size * region_size, emb_size, name,
                initializer, **kwargs)
        self._region_merge_fn = region_merge_fn
        self._word_emb = tf.get_variable(name + '_wordmeb', shape=[vocab_size, emb_size], 
                initializer=initializer)
        self._unit_id_bias = np.array([i * vocab_size for i in range(region_size)])
        self._region_size = region_size

    def _region_aligned_units(self, seq):
        """
        _region_aligned_unit
        """
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_seq = region_aligned_seq + self._unit_id_bias
        region_aligned_unit = super(ContextWordRegionEmbeddingLayer, self)._forward(region_aligned_seq)
        return region_aligned_unit
    
    def _forward(self, seq):
        """forward
        """
        region_radius = self._region_size / 2
        word_emb = tf.nn.embedding_lookup(self._word_emb, \
                tf.slice(seq, \
                [0, region_radius], \
                [-1, tf.cast(seq.get_shape()[1] - 2 * region_radius, tf.int32)]))
        word_emb = tf.expand_dims(word_emb, 2)
        region_aligned_unit = self._region_aligned_units(seq)
        embedding = region_aligned_unit * word_emb
        embedding = self._region_merge_fn(embedding, axis=2)
        return embedding


def main():
    """main"""
    pass

if '__main__' == __name__:
    main()

