#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: 
"""

from .layer import func_layer
from .layer import Layer

from .fc_layer import FCLayer
FCLayer = func_layer(FCLayer)

from .window_alignment_layer import WindowAlignmentLayer
WindowAlignmentLayer = func_layer(WindowAlignmentLayer)

from .embedding_layer import EmbeddingLayer
EmbeddingLayer = func_layer(EmbeddingLayer)

from .embedding_layer import WordContextRegionEmbeddingLayer
WordContextRegionEmbeddingLayer = func_layer(WordContextRegionEmbeddingLayer)

from .embedding_layer import ContextWordRegionEmbeddingLayer
ContextWordRegionEmbeddingLayer = func_layer(ContextWordRegionEmbeddingLayer)

from .vsum_layer import VSumLayer
VSumLayer = func_layer(VSumLayer)

from .vsum_layer import WeightedVSumLayer
WeightedVSumLayer = func_layer(WeightedVSumLayer)

from .concat_layer import ConcatLayer
ConcatLayer = func_layer(ConcatLayer)

