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

class VSumLayer(Layer):
    """VSumLayer"""
    def __init__(self, name='vsum', **args):
        Layer.__init__(self, name, **args)

    def _forward(self, x):
        return tf.reduce_sum(x, 1)


class WeightedVSumLayer(VSumLayer):
    """VSumLayer"""
    def _forward(self, emb, weight):
        weighted_emb = emb * weight
        return tf.reduce_sum(weighted_emb, 1)
