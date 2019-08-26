# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:40:57 2019

@author: Dekel
"""
from hpyutils import setattrs, rsetattr
from kerasExperiments import run_multi_exps_configs, BaseExp, createMultipleConfigExps
from kerasClassify import evaluate_mlp_model
import numpy as np
import pandas as pd
#################### Common Enron Experiments Infra ####################
class EnronBaseExp(BaseExp):
    def __init__(self,testgroupby):
        super().__init__()
        dsi = self.dataset_info
        dsi.num_runs = 1
        dsi.read_df_feature = True # Read pickled Spark dataset and extract features differently than default_exp
        
        setattrs(dsi.preprocess,
             text_cols = [ 'subj', 'body' ], # , 'people_format' # Important: Not used in old get_ngrams_data (.tsv) 'tok_to', 'tok_cc'
             use_filtered = True,
             filtered_prefix = 'filt_',     
        )
        dsi.metrics.testgroupby = testgroupby
        # Debug: Remove/Change
        dsi.csvEmailsFilePath =  "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/group_0.pkl"  # "D:/Dekel/Data/Text_py/Datasets/enron_deriv/sender_pkls/v2_vec_glv100_lda50_w2v100_ngr2_dct_ner/cache/group_0.pkl/small.parquet"
        dsi.labels_map = None
        dsi.sub_sample_mapped_labels = None
        
    def tag_metrics(self,dataset_info,df_test_metrics):
        super().tag_metrics(dataset_info,df_test_metrics)
        df_test_metrics['text_cols'] = ' '.join(dataset_info.preprocess.text_cols)
        df_test_metrics['csvEmailsFilePath'] = dataset_info.csvEmailsFilePath


# arrDctParams - Array of dctParams to apply to base config
# TODO: tag_metrics - by the keys in the dicts (col for each key)
def createEnronMultipleConfigExps(testgroupby, arrDctParams):
    return createMultipleConfigExps(arrDctParams,lambda : EnronBaseExp(testgroupby))


############################################################ Sepcific Enron Exps ################################################################
    
###################### Features lift (compare metrics with 2 groups of features) ######################

dctSubjBody = { 
  'preprocess.text_cols' : [ 'subj', 'body' ],
}

dctSubjectBody = { 
  'preprocess.text_cols' : [ 'subject', 'content' ],
}


#dctSubjBodyToCC = { 
#  'preprocess.text_cols' : [ 'subj', 'body','tok_to','tok_cc' ],
##   'train.classifier_func' : evaluate_mlp_model
#}
# exps = createEnronMultipleConfigExps('sender',[dctSubjBody,dctSubjectBody]) #dctSubjBodyToCC,

###################### Randomized Labels Test ######################
def randomizeLabels(dataset_info):
    dataset_info.ds.df['lbl_rand'] = np.random.choice(a=[0, 1], size=len(dataset_info.ds.df))
    dataset_info.ds.label_col_name = 'lbl_rand'
    
#dctRandLblParams = { 
# 'hooks.afterFeatures' : randomizeLabels,
#}    
#exps = createEnronMultipleConfigExps('sender',[dctRandLblParams])
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

#dctFilterFewEmailGroupParams = { 
# 'hooks.afterFeatures' : createfilterFewEmailsGroup('sender',5),
#}    
#exps = createEnronMultipleConfigExps('sender',[dctFilterFewEmailGroupParams] )


###################### W2V Trained Embeddings  ######################
def concatW2vTrainedWithBOW(df,dataset_info):     
    '''
    adds 'features' coludfmns by combining feature vectors of subject, body, different embeddings, etc
    '''    
    # Inside lambda - Convert each DenseVector cell (subj,body) to np.arr of np.arr --> elementwise avg of 2 vectors (avg of arr of arr) using np --> tolist (features is a list)
    
    # BOW + W2V_100
    df['features'] = df.apply(lambda row: np.average(np.array([list(row['emb_w2v_body']),list(row['emb_w2v_subj'])]),axis=0).tolist() + row['features'].tolist(), axis=1)    
    
    return df

dctGetFeatureVectorParams = {
  'load.required_cols' : ['emb_w2v_subj','emb_w2v_body'],      
  'preprocess.modifyFeatureVector' : concatW2vTrainedWithBOW,
  'preprocess.select_best' : None,
}    

# exps = createEnronMultipleConfigExps('sender',[dctGetFeatureVectorParams])

# exps = [EnronBaseExp('sender')] # Exp with default enron config (no dctParams)


###################### Glove Pretrained Embeddings  ######################
def getAggListVecs(lstVecs):
    return np.average(np.array([np.array(v) for v in lstVecs]),axis=0).tolist()
    
def concatGlovePretrainedWithBOW(df,dataset_info):
    '''
    adds 'features' coludfmns by combining feature vectors of subject, body, different embeddings, etc
    '''    
    # Inside lambda - Convert each DenseVector cell (subj,body) to np.arr of np.arr --> elementwise avg of 2 vectors (avg of arr of arr) using np --> tolist (features is a list)
    
    
    
    # BOW + Glove
    df['features'] = df.apply(lambda row: getAggListVecs(row['emb_glv_body'] + row['emb_glv_subj']) , axis=1) # + row['features'].tolist()
    
    # Filter out Nan (No glove vectors at all) from exp
    new_filter_col_name = 'remove_empty_glv'
    df[new_filter_col_name] = False
    df.loc[df['features'].apply(lambda lst: type(lst) != list),new_filter_col_name] = True
    dataset_info.ds.setFilterCol(new_filter_col_name)        
    return df

dctGloveParams = { 
  'load.required_cols' : ['emb_glv_subj','emb_glv_body'],
  'preprocess.modifyFeatureVector' : concatGlovePretrainedWithBOW,
  'preprocess.select_best' : None,
}    

exps = createEnronMultipleConfigExps('sender',[dctGloveParams])
########################################## Run multi exp ################################################    
df_results = run_multi_exps_configs(exps)

out_df = df_results[['accuracy','precision','recall','sel_tpr','roc_auc', 'text_cols']].groupby('text_cols').mean()
print(out_df)
