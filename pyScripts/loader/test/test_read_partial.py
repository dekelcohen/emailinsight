# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:46:05 2019

@author: Dekel
"""
from ..read_partial import readDF

############# Unit Test ####################
my_min_cols = ['conversationId', 'createdDateTime', 'folderName', 'id', 'inferenceClassification', 'internetMessageId', 
 'sentDateTime', 'subject', 'userId', 'sender', 'body', 'subj', 'PartId', 'label', 'group', 'to_rcpt', 'cc_rcpt']


my_data_path = 'D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/newnew.pkl'
my_req_cols =  [ 'emb_glv_subj', 'ner_body','emb_w2v_body'] #'emb_glv_body',
df = readDF(my_data_path,my_req_cols,my_min_cols)

print(df.columns)