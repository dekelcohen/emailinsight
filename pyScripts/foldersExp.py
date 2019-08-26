# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:12:42 2019

@author: Dekel
"""
from hpyutils import MyObj, setattrs, rsetattr
from kerasExperiments import run_multi_exps_configs, BaseExp, createMultipleConfigExps
from kerasClassify import evaluate_mlp_model

class EnronFoldersBaseExp(BaseExp):
    def __init__(self):
        super().__init__()
        dsi = self.dataset_info
        dsi.num_runs = 1       
        
        setattrs(dsi.preprocess,
             text_cols = [ 'subject', 'content', 'sender','to','cc' ], # , 'people_format' # Important: Not used in old get_ngrams_data (.tsv) 'tok_to', 'tok_cc'
        
        )
        
              
        # dsi.csvEmailsFilePath =  "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/group_0.pkl"  # "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/cache/group_0.pkl/small.parquet"
        
        
    def tag_metrics(self,dataset_info,df_test_metrics):
        super().tag_metrics(dataset_info,df_test_metrics)
        df_test_metrics['text_cols'] = ' '.join(dataset_info.preprocess.text_cols)
        df_test_metrics['csvEmailsFilePath'] = dataset_info.csvEmailsFilePath
        

def createEnronFoldersConfigExps(arrDctParams):
    return createMultipleConfigExps(arrDctParams,lambda : EnronFoldersBaseExp())


################# Default Folders Exp ##########################
exps = createEnronFoldersConfigExps([{ 'train.classifier_func' : evaluate_mlp_model}] )
df_results = run_multi_exps_configs(exps)

out_df = df_results[['accuracy','precision','recall','sel_tpr','roc_auc', 'text_cols']].groupby('text_cols').mean()
# out_df = df_results[['accuracy','precision','recall','sel_tpr','roc_auc']]
print(out_df)



