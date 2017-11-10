#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: 
"""

import numpy as np
import tensorflow as tf
from .layer import Layer

class WindowAlignmentLayer(Layer):
    """WindowAlignmentLayer"""
    def __init__(self, win_size, name="WinAlig", **args):
        Layer.__init__(self, name, **args)
        self._win_size = win_size

    def _forward(self, x):
        """forward
            win_size: window size
        """
        win_radius = self._win_size / 2
        aligned_seq = map(lambda i: tf.slice(x, [0, i - win_radius], [-1, self._win_size]), \
                xrange(win_radius, x.shape[1] - win_radius))
        aligned_seq = tf.convert_to_tensor(aligned_seq)
        aligned_seq = tf.transpose(aligned_seq, perm=[1, 0, 2])
        return aligned_seq
