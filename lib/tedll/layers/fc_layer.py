#!/usr/bin/env python
#-*- coding: gb18030 -*-

"""
Author	:

Date	:

Brief	: 
"""

import numpy as np
import tensorflow as tf
from .layer import Layer

class FCLayer(Layer):
    """FCLayer"""
    def __init__(self, in_size, out_size, name="fc", initializer=None, with_bias=True, **kwargs):
        Layer.__init__(self, name, **kwargs)
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._W = self.get_variable(name + '_W', shape=[in_size, out_size],\
                initializer=initializer)
        self._b = None
        if with_bias:
            self._b = tf.get_variable(name + '_b', initializer=tf.zeros([out_size]))

    def _forward(self, x):
        if self._b is not None:
            y = tf.nn.xw_plus_b(x, self._W, self._b)
        else:
            y = tf.matmul(x, self._W)
        return y


class SeqFCLayer(FCLayer):
    """SeqFCLayer"""
    def __init__(self, in_size, out_size, name="seqfc", **kwargs):
        super(SeqFCLayer, self).__init__(in_size, out_size, name, **kwargs)
        self._in_size = in_size
        self._out_size = out_size

    def _forward(self, x):
        xs = tf.shape(x)
        h = tf.reshape(x, [-1, self._in_size])
        h = super(SeqFCLayer, self)._forward(h)
        h = tf.reshape(h, [-1, xs[1], self._out_size])
        return h

