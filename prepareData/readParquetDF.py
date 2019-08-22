# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:49:00 2019

@author: Dekel
"""

# import pyarrow.parquet as pq
 
# all_emails_kaminskiv_parquet =  pq.read_table("D:/Dekel/Data/Text_py/Datasets/enron_deriv/test_pq/part-00000-tid-1847770169448339641-75e1f9fb-1e84-4c2d-81ef-c19461a586e2-0-1-c000.snappy.parquet") # all_emails_kaminskiv.parquet - failed
# emails_df = all_emails_kaminskiv_parquet.to_pandas()


from pathlib import Path
import pandas as pd

#df = pd.read_pickle('D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/group_data_1.pkl')
# df_pk.head()

# Failed parquet reading attempt 
#data_dir = Path('D:/Dekel/Data/Text_py/Datasets/enron_deriv/test_pq')
#full_df = pd.concat(
#    pd.read_parquet(parquet_file)
#    for parquet_file in data_dir.glob('*.parquet')
#)


#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.tree import DecisionTreeClassifier
#import pandas as pd
#import numpy as np
#df = pd.DataFrame({'tweets': ['one', 'two', 'one two three', 'four'], 'labels': [1, 0, 1, 0]})
#vectoriser = TfidfVectorizer()
#df['tweetsVect'] = list(vectoriser.fit_transform(df['tweets']).toarray())
#tree = DecisionTreeClassifier()
#tree.fit(df['tweetsVect'].tolist(), df['labels'].tolist())

#import numpy as np
#import matplotlib.pyplot as plt
#
#logbins = np.logspace(0, 16, num=17, base=2) # 1,2,4,8 .... 2^16
#linbins = np.linspace(0, 100, num=49)
#cbins = [0, 1, 2,3,4,5,6,7,8,10, 20, 50,100, max(df_final_stats['train_count'])]
#x = list(df_final_stats['train_count'])
#plt.figure()
#plt.hist(x, bins=cbins)
## plt.xscale('log')
#plt.show()


#testgroup = 'sender'
#df = pd.read_pickle('D:/Dekel/Data/Text_py/Datasets/enron_deriv/df.pkl')
#df_t = pd.read_pickle('D:/Dekel/Data/Text_py/Datasets/enron_deriv/df_t.pkl')
#
## Assign predictions, only to rows selected to test set
## predictions[0] corresponds to first row in get_X_test() --> with row_index 5001 ...
#
#
#df_t['correct_pred'] = y_true == df_t['predictions']
#
#df_stat = pd.DataFrame()
#df_stat[testgroup] = list(df_t.groupby(testgroup).groups.keys())
#df_stat['grp_avg_acc'] = list(df_t.groupby(testgroup)['correct_pred'].mean())
#df_stat['test_count'] = list(df_t.groupby(testgroup)[testgroup].count())
#df_train_groups = pd.DataFrame()
#df_train_groups[testgroup] = list(df.groupby(testgroup).groups.keys())
#df_train_groups['train_count'] = list(df.groupby(testgroup)[testgroup].count())
#
#df_final_stats =  pd.merge(df_stat,df_train_groups,on=testgroup,how='left').fillna(0)
#



import pandas as pd
df_pk = pd.read_pickle("D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/group_0.pkl")
# df = df_pk.head(300)
#del df_pk
#import gc
#gc.collect()


import pickle
picklefilePath = "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/models/text_Word2VecModel_2.pkl"
with open(picklefilePath,'rb') as fp:
    w2v_df = pickle.load(fp)
    
    
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

w2v_df['np_vec'] = w2v_df.apply(lambda row: row['vector'].tolist() , axis=1)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
mat_vec = np.array(w2v_df['np_vec'].tolist())
tsne_results = tsne.fit_transform(mat_vec)
w2v_df['tsne-2d-one'] = list(tsne_results[:,0])
w2v_df['tsne-2d-two'] = list(tsne_results[:,1])

fig, ax = plt.subplots(figsize=(16,10))
# ax.scatter(tsne_results[0:2,0],tsne_results[0:2,1])

# plt.figure(figsize=(16,10))

sns.scatterplot(
    x="tsne-2d-one", 
    y="tsne-2d-two",    
    data=w2v_df,
    legend="full",
    alpha=0.3
)

 w2v_df.apply(lambda row: ax.annotate(row['word'], (row['tsne-2d-one'], row['tsne-2d-two'])), axis=1)
# w2v_df.apply(lambda row: ax.text(row['tsne-2d-one'], row['tsne-2d-two'], row['word']), axis=1)
# ax.text(point['x']+.02, point['y'], str(point['val']))

for i in range(0,len(w2v_df) - 1):
    row = w2v_df.iloc[i]
    plt.annotate(row['word'], (row['tsne-2d-one'], row['tsne-2d-two']))
    
##################### Our similar to Good sample ######################
n_pts = 6000
y = list(tsne_results[0:n_pts,1])
z = list(tsne_results[0:n_pts,0])
n = [lbl.replace("$","Dlr") for lbl in list(w2v_df['word'])[0:n_pts]]


fig, ax = plt.subplots(figsize=(80,64))
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))    
################### Good sample ################################    
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 123]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))