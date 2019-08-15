# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:40:57 2019

@author: Dekel
"""
from hpyutils import setattrs, rsetattr
from kerasExperiments import run_multi_exps_configs, BaseExp
import numpy as np
import pandas as pd
#################### Common Enron Experiments Infra ####################
class EnronBaseExp(BaseExp):
    def __init__(self,testgroupby):
        super().__init__()
        dsi = self.dataset_info
        dsi.num_runs = 3
        dsi.read_exp_pkl = True # Read pickled Spark dataset and extract features differently than default_exp
        
        setattrs(dsi.preprocess,
             text_cols = [ 'subject', 'body' ], # , 'people_format' # Important: Not used in old get_ngrams_data (.tsv) 'tok_to', 'tok_cc'
             use_filtered = True,
             filtered_prefix = 'filt_',     
        )
        dsi.metrics.testgroupby = testgroupby
        # Debug: Remove/Change
        dsi.csvEmailsFilePath =  "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/group_data_0.pkl" 
        dsi.labels_map = None
        dsi.sub_sample_mapped_labels = None
        
    def tag_metrics(self,dataset_info,df_test_metrics):
        super().tag_metrics(dataset_info,df_test_metrics)
        df_test_metrics['text_cols'] = ' '.join(dataset_info.preprocess.text_cols)
        df_test_metrics['csvEmailsFilePath'] = dataset_info.csvEmailsFilePath


# TODO: Support multiple keys combinatorics in a single dict or array of dctParams (simpler, no combinatorics, full user control)
# TODO: tag_metrics - by the keys in the dicts (col for each key)
def createEnronMultipleConfigExps(testgroupby, dctParams):
    '''
    Return a list of experiments with different params, according to dctParams
    '''
    if len(dctParams.keys()) != 1:
        raise Exception('createEnronMultipleConfigExps: dctParams must contain exactly 1 key with a value / array of values. TODO: Support multiple keys combinatorics in dctParams')
    exps = []
    prmKey = list(dctParams.keys())[0]
    vals = dctParams[prmKey]
    if type(vals) != list:
        vals = [vals]
    for val in vals:
        exp = EnronBaseExp(testgroupby)
        rsetattr(exp.dataset_info,prmKey,val)        
        exps.append(exp)
    return exps

############################################################ Sepcific Enron Exps ################################################################
    
###################### Features lift (compare metrics with 2 groups of features) ######################

#dctTextColsParams = { 
#  'preprocess.text_cols' : [[ 'subject', 'body' ],[ 'subject', 'body','tok_to','tok_cc' ]]  
#}
#exps = createEnronMultipleConfigExps('sender',dctTextColsParams)

###################### Randomized Labels Test ######################
def randomizeLabels(dataset_info):
    dataset_info.ds.df['lbl_rand'] = np.random.choice(a=[0, 1], size=len(dataset_info.ds.df))
    dataset_info.ds.label_col_name = 'lbl_rand'
    
#dctRandLblParams = { 
# 'hooks.afterFeatures' : randomizeLabels,
#}    
#exps = createEnronMultipleConfigExps('sender',dctRandLblParams)
#

###################### Filter senders with small count of training sample  ######################
def createfilterFewEmailsGroup(testgroup,min_sender_emails_to_keep):
    def filterFewEmailsGroup(dataset_info):        
        # Work on entire df (not df.get_X_train) -->  Filter few email senders from both train and test, regardless of wethere some of their emails were already filtered         
        df = dataset_info.ds.df
        stats_df = df.groupby(testgroup).agg({testgroup : 'count'})
        
        filt_groups_df = stats_df[stats_df[testgroup] < min_sender_emails_to_keep].drop(columns=[testgroup])
        new_filter_col_name = 'filter_few_rows_group'
        filt_groups_df[new_filter_col_name] = True
        filt_df = pd.merge(df,filt_groups_df,left_on=testgroup,right_index=True,how='left')
        filt_df[new_filter_col_name].fillna(False,inplace=True)
                
        # copy df[dataset_info.ds.filer_col_name] and logic OR with our list of True for emails of senders to filter out
        dataset_info.ds.setFilterCol(new_filter_col_name)        
        dataset_info.ds.df = filt_df
        print('************** filterFewEmailsGroup:  filtered %d rows (emails) of %d %s' % (sum(filt_df[new_filter_col_name]),len(filt_groups_df),testgroup))    
        
    return filterFewEmailsGroup

dctFilterFewEmailGroupParams = { 
 'hooks.afterFeatures' : createfilterFewEmailsGroup('sender',5),
}    
exps = createEnronMultipleConfigExps('sender',dctFilterFewEmailGroupParams )


########################################## Run multi exp ################################################    
df_results = run_multi_exps_configs(exps)

out_df = df_results[['accuracy','precision','recall','sel_tpr','roc_auc', 'text_cols']].groupby('text_cols').mean()
print(out_df)
