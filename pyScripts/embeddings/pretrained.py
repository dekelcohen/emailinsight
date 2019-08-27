# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:18:45 2019

@author: Dekel
"""

import gensim.downloader as api
import numpy as np

modelCache = dict()
            
def loadGloveGensimDownloader(dims = 100):    
    model_key = "glove-wiki-gigaword-" + str(dims)
    if not model_key in modelCache:
        modelCache[model_key] = api.load(model_key)            
    return modelCache[model_key]


def getAggListVecs(lstVecs):
    return np.average(np.array([np.array(v) for v in lstVecs]),axis=0).tolist()

def getWordVectorsFromOneHot(vec_features, feature_names,w2v):
    '''
    vec_features : sparse feature (OneHot like) vector - 0 if no word and > 0 if word 
    feature_names - np.array of [feature_tok1,feature_tok2,...]
    w2v - keyedVector with word-->vector mapping - https://radimrehurek.com/gensim/models/keyedvectors.html
    '''
    words = feature_names[vec_features > 0]
    vecList = [getWordVec(word,w2v) for word in words]
    return vecList
    
    
def getWordVec(word,w2v):
    if word in w2v:
        return w2v[word]
    else:
        return np.zeros(w2v['yes'].shape[0])

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

w2v = loadGloveGensimDownloader(100)
# getWordVec('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabb',w2v)  
        
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









