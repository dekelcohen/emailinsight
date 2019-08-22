# -*- coding: utf-8 -*-
"""
Dsiplay a scatter plot of word vectors (labeled by their word text) to manually examine word2vec training results
Created on Thu Aug 22 18:03:29 2019

@author: Dekel
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle


# Load word2vec df from the below path (.pkl) 'word' (string), 'vector' (DenseVector) columns
picklefilePath = "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/models/text_Word2VecModel_2.pkl"
with open(picklefilePath,'rb') as fp:
    w2v_df = pickle.load(fp)
    
w2v_df['np_vec'] = w2v_df.apply(lambda row: row['vector'].tolist() , axis=1)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
mat_vec = np.array(w2v_df['np_vec'].tolist())
tsne_results = tsne.fit_transform(mat_vec)

n_pts = 6000
y = list(tsne_results[0:n_pts,1])
z = list(tsne_results[0:n_pts,0])
n = [lbl.replace("$","Dlr") for lbl in list(w2v_df['word'])[0:n_pts]]


fig, ax = plt.subplots(figsize=(80,64))
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))    