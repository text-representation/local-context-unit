#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: Universal Config
"""

import json

class Config(object):
    """Config"""
    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)

        if config:
            self._update(config)

    def add(self, key, value):
        """add

        Args:
                key(type):
                value(type):
        Returns:
                type:
        """
        self.__dict__[key] = value
        
    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in config[key]]
            
        self.__dict__.update(config)

    def __repr__(self):
        return '%s' % self.__dict__


def main():
    """unit test for main"""
    config = Config(config_file='./conf/model.config')
    print config.train.max_epoch


if '__main__' == __name__:
    main()

