# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""
import functools


def setattrs(_self, **kwargs):
    for k,v in kwargs.items():
        setattr(_self, k, v)


# Generic object
class MyObj():
    def __init__(self):
        pass


# rgetattr and rsetattr are drop-in replacements for getattr and setattr, which can also handle dotted attr strings.
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
