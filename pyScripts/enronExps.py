# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:40:57 2019

@author: Dekel
"""
from hpyutils import setattrs
from kerasExperiments import run_multi_exps_configs, BaseExp

#################### Common Enron Experiments Infra ####################
class EnronBaseExp(BaseExp):
    def __init__(self,testgroupby):
        super().__init__()
        dsi = self.dataset_info
        dsi.num_runs = 1
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


def createEnronMultipleFeaturesExps(testgroupby, text_cols_grps):
    exps = []
    for text_cols in text_cols_grps:
        exp = EnronBaseExp(testgroupby)
        exp.dataset_info.preprocess.text_cols = text_cols
        exps.append(exp)
    return exps

################ Sepcific Enron Exps ####################
    
# Features lift (compare metrics with 2 groups of features)
exps = createEnronMultipleFeaturesExps('sender',[[ 'subject', 'body' ],[ 'subject', 'body','tok_to','tok_cc' ]])

df_results = run_multi_exps_configs(exps)

out_df = df_results[['accuracy','precision','recall','sel_tpr','roc_auc', 'text_cols']].groupby('text_cols').mean()
print(out_df)
