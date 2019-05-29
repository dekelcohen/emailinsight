# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:49:21 2019

@author: Dekel
"""



def setattrs(_self, **kwargs):
    for k,v in kwargs.items():
        setattr(_self, k, v)

