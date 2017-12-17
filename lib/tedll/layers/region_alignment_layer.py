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

class RegionAlignmentLayer(Layer):
    """RegionAlignmentLayer"""
    def __init__(self, region_size, name="RegionAlig", **args):
        Layer.__init__(self, name, **args)
        self._region_size = region_size

    def _forward(self, x):
        """forward
            region_size: region size
        """
        region_radius = self._region_size / 2
        aligned_seq = map(lambda i: tf.slice(x, [0, i - region_radius], [-1, self._region_size]), \
                xrange(region_radius, x.shape[1] - region_radius))
        aligned_seq = tf.convert_to_tensor(aligned_seq)
        aligned_seq = tf.transpose(aligned_seq, perm=[1, 0, 2])
        return aligned_seq
