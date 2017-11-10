#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: 
"""

import sys
import tensorflow as tf
from functools import wraps

class Layer(object):
    """Layer"""
    def __init__(self, name, activation=None, dropout=None, decay_mult=None):
        self._name = name
        self._activation = activation
        self._dropout = dropout
        self._decay_mult = decay_mult

    def get_variable(self, name, **kwargs):
        """get variable with regularization'
        """
        if self._decay_mult:
            kwargs['regularizer'] = lambda x: tf.nn.l2_loss(x) * self._decay_mult
        return tf.get_variable(name, **kwargs)

    def __call__(self, *inputs):
        """forward

        Args:
                inputs(type): input op
        Returns:
                type: output op
        """
        
        outputs = []
        for x in inputs:
            if type(x) == tuple or type(x) == list:
                y = self._forward(*x)
            else:
                y = self._forward(x)
            if self._activation:
                y = self._activation(y)
            if self._dropout:
                if hasattr(tf.flags.FLAGS, 'training'):
                    y = tf.cond(tf.flags.FLAGS.training, 
                            lambda: tf.nn.dropout(y, keep_prob = 1.0 - self._dropout), 
                            lambda: y)
                else:
                    y = tf.nn.dropout(y, keep_prob = 1.0 - self._dropout)
            outputs.append(y)
        
        def get_shape_desc(x):
            """get shape description
            """
            if type(x) == list or type(x) == tuple:
                return '[%s]' % ', '.join([str(xi.shape) for xi in x])
            return str(x.shape)
        print >> sys.stderr, '''layer {
    type: %s
    name: %s
    shape[in]: %s
    shape[out]: %s
}''' % (self.__class__.__name__, self._name, get_shape_desc(x), get_shape_desc(y))
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def _forward(self, x):
        return x


def func_layer(cls):
    """layer funcition wrapper
    """
    def is_tensor(x):
        """is_tensor
        """
        tx = type(x)
        if tx == tuple or tx == list:
            for xi in x:
                if type(xi) == tf.Tensor:
                    return True
            return False
        return tx == tf.Tensor
    @wraps(cls)
    def _deco(*args, **kwargs):
        if len(args) > 0:
            idx = len(args)
            for i, arg in enumerate(args):
                if not is_tensor(arg):
                    idx = i
                    break
            if idx == 0:
                return cls(*args, **kwargs)
            return cls(*args[idx:], **kwargs)(*args[:idx])
        else:
            return cls(**kwargs)
        
    return _deco


