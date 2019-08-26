# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:46:05 2019

@author: Dekel
"""
import os
import pandas as pd
import shutil

def getPathComponents(fullPath):
    _, file_extension = os.path.splitext(fullPath)
    folder = os.path.dirname(fullPath)    
    full_filename = fullPath.replace(os.path.dirname(fullPath),'').strip("/")
    return folder, full_filename, file_extension
    
def readDF(dataFilePath,req_cols,min_cols, verbose = True):
    '''
    Read a dataframe from disk, using small df (min set of columns) to save memory and runtime
    '''    
    folder, full_filename, file_extension = getPathComponents(dataFilePath)
    cacheFolder = os.path.join(folder,'cache',full_filename)
    minDataFilePath = os.path.join(cacheFolder,'min.pkl')
    
    
    # invalidate cache, If file at dataFilePath is newer than its cache folder timestamp
    if os.path.exists(cacheFolder) and os.path.getmtime(cacheFolder) < os.path.getmtime(dataFilePath):
         shutil.rmtree(cacheFolder)
         if verbose:
             print('Invalidate readDF cache at %s' % (cacheFolder))
    
    # Create cache folder for this df
    if not os.path.exists(cacheFolder):
        os.makedirs(cacheFolder)
        if verbose:
             print('Create readDF cache at %s' % (cacheFolder))
    
    
    # Create min (small) df file, with base cols (subj,body ...)
    if not os.path.exists(minDataFilePath):
        df = pd.read_pickle(dataFilePath)
        df_min = df[min_cols]
        df_min.to_pickle(minDataFilePath)
    else:
        df = pd.read_pickle(minDataFilePath)
    # For each column not in small / min df --> try to find a per column file to read and add to min_df. If file not found, fallback to reading the large file    
    extra_cols = set(req_cols) if req_cols else set() - set(min_cols)
    
    # Test if all extra_cols have a datafile
    all_extra_cols_have_files = True
    for ec in extra_cols:
        colDataFilePath = os.path.join(cacheFolder,ec+'.pkl')
        if not os.path.exists(colDataFilePath):
            all_extra_cols_have_files = False
            break
        
    if all_extra_cols_have_files:
         for ec in extra_cols:
            colDataFilePath = os.path.join(cacheFolder,ec+'.pkl')        
            col_df = pd.read_pickle(colDataFilePath)
            df[ec] = col_df[ec]
    else:
         # Read full df and create file for each extra col that is doesn't alreayd have one
         if len(df.columns) <= len(min_cols):
             df = pd.read_pickle(dataFilePath)
         for ec in extra_cols:
             if not ec in df.columns:
                raise Exception('Failed to locate column %s in full data file %s' % (ec,dataFilePath))
             colDataFilePath = os.path.join(cacheFolder,ec+'.pkl')        
             if not os.path.exists(colDataFilePath):     
                 col_df = df[[ec]]
                 col_df.to_pickle(colDataFilePath)                                              
        
    return df


############## Unit Test ####################
#my_min_cols = ['conversationId', 'createdDateTime', 'folderName', 'id', 'inferenceClassification', 'internetMessageId', 
# 'sentDateTime', 'subject', 'userId', 'sender', 'body', 'subj', 'PartId', 'label', 'group', 'to_rcpt', 'cc_rcpt']
#
#
#my_data_path = 'D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/group_0.pkl'
#my_req_cols =  [ 'emb_glv_subj', 'ner_body','emb_w2v_body'] #'emb_glv_body',
#df = readDF(my_data_path,my_req_cols,my_min_cols)

# print(df.columns)