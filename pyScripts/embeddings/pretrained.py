# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:18:45 2019

@author: Dekel
"""

import gensim.downloader as api

modelCache = dict()
            
def loadGloveGensimDownloader(dims = 100):    
    model_key = "glove-wiki-gigaword-" + str(dims)
    if not model_key in modelCache:
        modelCache[model_key] = api.load(model_key)            
    return modelCache[model_key]


#import gensim
#import numpy as np

#class WordVectorsHelper:
#    def __init__(self,wvPath):
#        self.wv = gensim.models.KeyedVectors.load_word2vec_format(wvPath, binary=True, limit=300000)
#        self.wv.init_sims(replace=True)    
#    def getVector(self,word):
#        if word in self.wv.vocab:
#            self.wv.vectors_norm[self.wv.vocab[word].index]
#        





####### UnitTest ######

# glove = loadGloveGensimDownloader(100)
  
# Gensim old iface      
# wvh = WordVectorsHelper("models/GoogleNews-vectors-negative300.bin.gz")

#def loadGloveGensim(glovePath):
#    return WordVectorsHelper(glovePath)



# Problem: Perf memory (2GB compared to Gensim 0.5 GB)
#def loadGloveFromPath(glovePath):    
#    with open(glovePath, "rb") as lines:
#        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
#               for line in lines}
#    return w2v









