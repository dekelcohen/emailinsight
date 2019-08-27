# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:07:26 2019

@author: Dekel
"""

'''
 Check if server is up using Semaphore / IPC with model-path 
 If server running - Goto #Consumer 
 
 If not - test if binary cache doens't exist --> create it if needed
 
 # Launch it without waiting + pause for 3 secs (or wait for IPC that it loaded) to allow it to read the file 
'''
model_bin_path = "models/gensim-glove-wiki-gigaword-100.bin"
#w2v.init_sims(replace=True) 
#w2v.save(model_bin_path)


## Server 
from gensim.models import KeyedVectors
from threading import Semaphore
model = KeyedVectors.load(model_bin_path, mmap='r')
model.vectors_norm = model.vectors  # prevent recalc of normed vectors
model.most_similar('stuff')  # any word will do: just to page all in
Semaphore(0).acquire()  # just hang until process killed



# Simple: If not - print a clear message and exit
# Launch server
proc = Popen([cmd_str], shell=True,
             stdin=None, stdout=None, stderr=None, close_fds=True)

## Consumer 
model = KeyedVectors.load(model_bin_path, mmap='r')
model.vectors_norm = model.vectors  # prevent recalc of normed vectors

