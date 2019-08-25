from __future__ import print_function

from collections import Counter
import numpy as np
import pandas as pd
import os
import time
import pprint
import pickle
import json

from keras.callbacks import RemoteMonitor
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from mboxConvert import parseEmails,parseEmailCSV, parseEmailsCSV,addColumnsCSV,getEmailStats,mboxToBinaryCSV
from kerasPlotter import Plotter
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense, Dropout, Activation, Flatten

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numbers

stopwords_lst = None

def get_stopwords_list():
    global stopwords_lst
    if not stopwords_lst is None:
        return stopwords_lst
    # Load stopwords (TODO: lang)
    with open('./data/stopwords/eng/stopwords.json',  encoding="utf8") as fp:
        stopwords_lst = set(json.load(fp))    
    return stopwords_lst


from mboxConvert import parseEmails,parseEmailsCSV,getEmailStats,mboxToBinaryCSV

def get_word_features(dataset_info,verbose=True, nb_words=5000, skip_top=0, maxlen=None, as_matrix=True, matrix_type='count',
                      label_cutoff=0.01, max_n=1):
    (totalWordsCount, fromCount, domainCount, labels) = getEmailStats(dataset_info.ds.df)
    if verbose:
        print('Creating email dataset with labels %s ' % str(labels))
        print('Label word breakdown:')
        total = 0
        for label in labels:
            count = sum(totalWordsCount[label].values())
            total += count
            print('\t%s:%d' % (label, count))
        print('Total word count: %d' % total)

    labelCounts = {label: 0 for label in labels}
    for index, email in dataset_info.ds.df.iterrows():
        labelCounts[email.label] += 1
    cutoff = int(len(dataset_info.ds.df) * label_cutoff)
    removed = 0
    for label in labels[:]:
        if labelCounts[label] < cutoff or label == 'Important' or label == 'Unread' or label == 'Sent':
            removed += 1
            labels.remove(label)
    labelNums = {labels[i]: i for i in range(len(labels))}
    if verbose:
        print('Found %d labels below count threshold of %d ' % (removed, cutoff))
    if verbose:
        print('Creating email dataset with labels %s ' % str(labels))
        print('Label email count breakdown:')
        total = 0
        for label in labels:
            print('\t%s:%d' % (label, labelCounts[label]))
        print('Total emails: %d' % sum([labelCounts[label] for label in labels]))

    texts = []
    emailLabels = []
    updateIds = []
    for index, email in dataset_info.ds.df.iterrows():
        if email.label not in labels:
            continue
        text = email.sender + " " + str(email.subject) + " "
        text += email.fromDomain
        if email.to:
            text += email.to + " "
            if getattr(dataset_info,'toccDomains', False): 
                text += email.toDomain + " "
        if email.cc:
            text += email.cc + " "
            if getattr(dataset_info,'toccDomains', False): 
                text += email.ccDomain + " "           
            
        text += email.content
        texts.append(text.replace('\n', '').replace('\r', ''))
        emailLabels.append(labelNums[email.label])
        updateIds.append(email.updateId)
    emailLabels = np.array(emailLabels)
    dataset_info.ds.df['label_num'] = emailLabels # unique labels, after cutoff (rare labels are not included in emailLables)
    tokenize_vectorize(texts,labels, dataset_info,verbose, nb_words, as_matrix, matrix_type, max_n)

def write_csv(csvfile, emails, verbose=True):
    emails.to_csv(csvfile, index=False, sep='\t')
    if verbose:
        print('Wrote CSV to %s' % csvfile)

def read_csv(csvfile,verbose=True):
    dataframe = pd.read_csv(csvfile,header=0)
    labels = dataframe[u'label'].tolist()
    updateIds = dataframe[u'updateId'].tolist()
    if verbose:
        print('Read CSV with columns %s'%str(dataframe.columns))
    dataframe.drop(u'label',inplace=True,axis=1)
    dataframe.drop(u'updateId',inplace=True,axis=1)
    if u'Unnamed: 0' in dataframe.columns:
        dataframe.drop(u'Unnamed: 0',inplace=True,axis=1)    
    feature_matrix = dataframe.values
    feature_names = dataframe.columns
    return updateIds, feature_matrix,labels,feature_names

def write_info(txtfile, label_names, verbose=True):
    with open(txtfile,'w') as writeto:
        writeto.write(','.join(label_names))

def read_info(txtfile,verbose=True):
    with open(txtfile,'r') as readfrom:
        label_names=readfrom.readline().split(',')
    return label_names

def write_sequences(txtfile, sequences, labels, verbose=True):
    with open(txtfile,'w') as writeto:
        for sequence,label in zip(sequences,labels):
            #lol random demarcation markers so fun amirite
            writeto.write(','.join([str(x) for x in sequence])+';;;'+str(label)+'\n')
    if verbose:
        print('Wrote txt with %d lines'%len(sequences))

def read_sequences(txtfile,verbose=True):
    sequences = []
    labels = []
    linesnum = 0
    with open(txtfile,'r') as readfrom:
        for line in readfrom:
            linesnum+=1
            parts = line.split(';;;')
            split = parts[0].split(',')
            if len(split)<=1:
                continue
            sequences.append(np.asarray(split))
            labels.append((int)(parts[1]))
    if verbose:
        print('Read txt with %d lines'%linesnum)
    return sequences,labels

    dataframe = pd.read_csv(csvfile,header=0)
    labels = dataframe[u'label'].tolist()
    if verbose:
        print('Read CSV with columns %s'%str(dataframe.columns))
    dataframe.drop('label',inplace=True,axis=1)
    feature_matrix = dataframe.as_matrix()
    return feature_matrix,labels

def init_randomness(dataset_info):
    if hasattr(dataset_info.state, 'index_random_seed') and len(dataset_info.random_seed) > dataset_info.state.index_random_seed:
        np.random.seed(dataset_info.random_seed[dataset_info.state.index_random_seed])

def get_idx_to_new_label_dict(dataset_info):
  '''
  Return a dict that maps label idx to label name 0-->Save 1 --> DontSave ...
  '''
  return dict(zip(list(range(0,len(dataset_info.new_label_names))), dataset_info.new_label_names))             

def auto_subsample_dataset(dataset_info):
    '''
    Reduce dataset to dataset_info.new_total_samples, taking the same number of samples from every class.
    '''
    sub_sample = dataset_info.ds.get_df()
    if not hasattr(dataset_info, 'new_total_samples') or dataset_info.new_total_samples is None or dataset_info.new_total_samples == 0:
        return
    unq_lbls = np.unique(sub_sample[getattr(dataset_info.ds,'label_col_name','label_num')], return_counts=True)
    n_samples_per_label = dataset_info.new_total_samples * (1 - dataset_info.test_split) / len(unq_lbls[0])
    if any(unq_lbls[1] < min(15,n_samples_per_label)):
        raise Exception("One of the classes doesn't have enough labels %d, unq_lbls[1]=%s" % (n_samples_per_label, ','.join(unq_lbls[1])))
    # Accumulate reduced dataset (trainset) in new_X while the number of samples from each label isn't higher than n_samples_per_label
    per_label_cnt = Counter()
    in_subsample_auto_indexes = []
    for index, row in sub_sample.iterrows():
        if per_label_cnt[row['label']] < n_samples_per_label:
            in_subsample_auto_indexes.append(index)#save the index
            per_label_cnt[row['label']] += 1

    dataset_info.ds.df["in_subsample_auto"] = [True if i in in_subsample_auto_indexes else None for i in range(len(dataset_info.ds.df))]
    dataset_info.ds.train_col_name = "in_subsample_auto"

def get_groupby_labels(dataset_info, sub_sample):
    label_col_name = getattr(dataset_info.ds, 'label_col_name', 'label_num')
    unq_lbls = np.unique(sub_sample[label_col_name])
    if (len(unq_lbls) < len(dataset_info.new_label_names)):
        raise Exception("Missing data for labels, add more folders for Save/dontSave")
    # find observation index of each class levels
    groupby_lbls = {}
    for ii, lbl_idx in enumerate(unq_lbls):
        lbl_samples_idxs = [idx for idx, val in enumerate(sub_sample[label_col_name]) if val == lbl_idx]
        groupby_lbls[lbl_idx] = lbl_samples_idxs
    return groupby_lbls

def set_sample_idxs(dataset_info, sub_sample, groupby_lbls, label_size, column_name):
    # Undersample each label (in a loop) according to sub_sample_mapped_labels (above)
    idx_to_new_label = get_idx_to_new_label_dict(dataset_info)
    under_sample_idxs = [] # Holds all idx of samples (from all labels), selected to keep
    for lbl_idx, lbl_samples_idxs in groupby_lbls.items():
        lbl_sample_size = label_size[idx_to_new_label[lbl_idx]]
        size = [lbl_sample_size if lbl_sample_size < len(lbl_samples_idxs) else len(lbl_samples_idxs)]
        init_randomness(dataset_info)
        lbl_under_sample_idxs = np.random.choice(lbl_samples_idxs, size=size, replace=False).tolist()
        under_sample_idxs += lbl_under_sample_idxs
        print(column_name, ': ', size, idx_to_new_label[lbl_idx])
    count = 0
    in_subsample_label_indexes = []
    for index, row in sub_sample.iterrows():
        if count in under_sample_idxs:
            in_subsample_label_indexes.append(index)
        count += 1
    dataset_info.ds.df[column_name] = [True if i in in_subsample_label_indexes else None for i in range(len(dataset_info.ds.df))]

def gcd_list(list):
    def gcd(a, b):
        while b > 0:
            a, b = b, a % b
        return a

    result = list[0]
    for i in list[1:]:
        result = gcd(result, i)
    return result

def subsample_testset_by_label_stratified(dataset_info):
    df = dataset_info.ds.get_df()
    # take sub_sample - only row not in train
    sub_sample = df[df[getattr(dataset_info.ds, 'train_col_name', 'train')].isnull()]
    groupby_lbls = get_groupby_labels(dataset_info, sub_sample)
    # Undersample each label (in a loop) according to sub_sample_mapped_labels (above)
    idx_to_new_label = get_idx_to_new_label_dict(dataset_info)
    sub_sample_mapped_labels_values = [*dataset_info.sub_sample_mapped_labels.values()]
    gcd = gcd_list(sub_sample_mapped_labels_values)
    ratio_map_labels = {k: v // gcd for k, v in dataset_info.sub_sample_mapped_labels.items()}
    num_unit = min([len(v) // ratio_map_labels[idx_to_new_label[k]] for k, v in groupby_lbls.items()])
    label_size = {k: num_unit * v for k, v in ratio_map_labels.items()}
    set_sample_idxs(dataset_info, sub_sample, groupby_lbls, label_size, 'test')

def subsample_trainset_by_label_stratified(dataset_info):
    '''
    Reduce each label in dataset to specified amount in dataset_info.sub_sample_mapped_labels ex: { 'Save': 60 ,'DontSave' : 600 }
    '''
    sub_sample = dataset_info.ds.get_df()   
    groupby_lbls = get_groupby_labels(dataset_info, sub_sample)
    set_sample_idxs(dataset_info, sub_sample, groupby_lbls, dataset_info.sub_sample_mapped_labels, 'train')
    #dataset_info.ds.train_col_name = 'train'

def subsample_dataset_by_label_stratified(dataset_info):
    if not hasattr(dataset_info, 'sub_sample_mapped_labels') or dataset_info.sub_sample_mapped_labels is None or len(dataset_info.sub_sample_mapped_labels) == 0:
        return
    subsample_trainset_by_label_stratified(dataset_info)
    subsample_testset_by_label_stratified(dataset_info)

def get_class_weight(dataset_info):
    if not hasattr(dataset_info, 'class_weight') or dataset_info.class_weight is None:
        return None
    
    new_label_to_idx = dict(zip(dataset_info.new_label_names, list(range(0,len(dataset_info.new_label_names)))))
    return { new_label_to_idx[lbl_name]:dataset_info.class_weight[lbl_name] for lbl_name in dataset_info.class_weight.keys()}

def create_get_new_label_idx(dataset_info, new_total_labels):
    init_randomness(dataset_info)
    permuted_old_label_idxs = np.random.permutation(len(dataset_info.label_names))
    dataset_info.permuted_old_label_idxs = permuted_old_label_idxs
    def get_new_label_idx(old_label_idx):
        permuted_old_label_idx = permuted_old_label_idxs[old_label_idx]
        return int(permuted_old_label_idx % new_total_labels)
    return get_new_label_idx

    
def auto_map_labels(dataset_info):
    '''
    auto map the new_total_labels - permute labels and map each group to a new label 
    '''
    new_total_labels = len(dataset_info.new_label_names)
    old_num_labels = len(dataset_info.label_names)    
    if not (old_num_labels / new_total_labels).is_integer():
        raise Exception("old_num_labels must be integer %f" % (old_num_labels))
        
    get_new_label_idx = create_get_new_label_idx(dataset_info, new_total_labels)
    dataset_info.ds.df['new_label_auto'] = list(map(get_new_label_idx, dataset_info.ds.df[getattr(dataset_info.ds,'label_col_name','label_num')]))
    dataset_info.ds.label_col_name = 'new_label_auto'

    return new_total_labels

def filter_out_labels(dataset_info):        
    filter = []
    if getattr(dataset_info, 'labels_map',False):
        for index, row in dataset_info.ds.df.iterrows():
            if row['label'] in dataset_info.labels_map \
                    and dataset_info.labels_map[row['label']] not in getattr(dataset_info, "labels_map_filter_names", []):
                filter.append(False)
            elif row['label'] in dataset_info.labels_map:
                filter.append(True)
            elif 'default_mapping' in dataset_info.labels_map \
                    and dataset_info.labels_map['default_mapping'] not in getattr(dataset_info, "labels_map_filter_names", []):
                filter.append(False)
            else:
                filter.append(True)
    else:
        filter = [False] * len(dataset_info.ds.df)
        
    dataset_info.ds.df["label_filter_out"] = filter
    dataset_info.ds.setFilterCol("label_filter_out")
    

def map_labels(dataset_info):
    '''
    Manually map labels according to user supplied labels_map
    '''
    # First map orig label idx (3) --> orig label name (ex: 'Inbox') --> new label name ('Save') --> new label idx
    orig_label_idx_to_name = dict(zip(list(range(0,len(dataset_info.label_names))),dataset_info.label_names))     
    new_label_to_idx = dict(zip(dataset_info.new_label_names, list(range(0,len(dataset_info.label_names)))))     
    def get_new_label_idx(idx_orig_label):
        orig_label = orig_label_idx_to_name[idx_orig_label] # 0 --> 'Inbox'
        # If not found in labels_map, try to use 'all_others' key if exist
        # Assumes label_col_name is the original label_num (folders)
        if orig_label in dataset_info.labels_map:
            new_label = dataset_info.labels_map[orig_label] # { 'Inbox' : 'Save','Notes inbox' : 'Save', 'default_mapping' : 'DontSave' }
        elif 'default_mapping' in dataset_info.labels_map:
            new_label = dataset_info.labels_map['default_mapping']
        else:
            raise Exception("Failed to map %s"%(orig_label))
        # set value None for labels in dataset_info.labels_map_filter_names ex: "Omit"
        new_label_idx = None
        if new_label in new_label_to_idx:
            new_label_idx = new_label_to_idx[new_label]
        return new_label_idx

    #sub_sample = dataset_info.ds.get_df(sorted=False)
    dataset_info.ds.df["new_label_map"] = list(map(get_new_label_idx, dataset_info.ds.df[getattr(dataset_info.ds,'label_col_name','label_num')]))
    dataset_info.ds.label_col_name = "new_label_map"
    new_total_labels = len(dataset_info.new_label_names)
    return new_total_labels


def simple_train_test_split(dataset_info):
    if not getattr(dataset_info,'test_split',0) > 0:
        return
    df = dataset_info.ds.df.sort_values(by=['index_row'])    
    idx_row_train_start = int(dataset_info.test_split*len(df))
    df['train'] = df.apply (lambda row: True if row['index_row'] >= idx_row_train_start else None, axis=1)
    df['test'] = df.apply (lambda row: True if row['index_row'] < idx_row_train_start else None, axis=1)    
    dataset_info.ds.df = df
    
 
def make_dataset(dataset_info):
    '''
    Split train, test subsample and remap labels
    '''                      
    num_labels = len(dataset_info.label_names)
    num_examples = dataset_info.ds.get_df().shape[0]
    init_randomness(dataset_info)
    random_order = np.random.permutation(num_examples)
    # Subsample dataset to get dataset_info.new_total_samples
    dataset_info.ds.df["index_row"] = [int((np.where(random_order == i))[0][0]) for i in range(len(dataset_info.ds.df))]
    #auto_subsapmle_dataset --> remove X_train, Y_train, send subdataset by get_train_set(dataset_info.ds.df)
    filter_out_labels(dataset_info)
    auto_subsample_dataset(dataset_info=dataset_info)
    # filter out row labels that define "Omit"
    # Map labels (ex: from folders to binary 2 folder groups)
    if getattr(dataset_info,'new_label_names',False) and not hasattr(dataset_info,'labels_map'):
        num_labels = auto_map_labels(dataset_info)
    # Map labels manually    
    if dataset_info.labels_map:
        num_labels = map_labels(dataset_info)
    # Subsample manually according to [optional] sub_sample_mapped_labels.     
    subsample_dataset_by_label_stratified(dataset_info)

    # Simple train/test split: If none of the above subsample by labels methods created train and test columns - try to create it now
    if not 'train' in dataset_info.ds.df:
        simple_train_test_split(dataset_info)
            
    return num_labels

def get_emails(emailsFilePath,verbose=True):
    picklefile = 'pickled_emails.pickle'
    if os.path.isfile(picklefile):
        with open(picklefile,'rb') as load_from:
            emails = pickle.load(load_from)
    else:
        # Uncomment to parse .mbox exported from Gmail
        # emails = parseEmails('.',printInfo=verbose)
        # Uncomment to parse CSV 
        emails = parseEmailsCSV(emailsFilePath)
        with open(picklefile,'wb') as store_to:
            pickle.dump(emails,store_to)    
    return emails

def convert_emails(df):
    addColumnsCSV(df)


    

def tokenize_vectorize(texts,labels, dataset_info,verbose=True, nb_words=5000, as_matrix=True, matrix_type='count',
                       max_n=1):
    '''
    Tokenize and Featurize from text --> tokens --> Count/TFIDF Vectorizers (optionally removing stopwords)
    '''
    if getattr(dataset_info,'use_keras_tokenizer',False) and max_n == 1 or not as_matrix:
        print('Using Keras Tokenizer')
        tokenizer = Tokenizer(nb_words)
        tokenizer.fit_on_texts(texts)
        reverse_word_index = {tokenizer.word_index[word]: word for word in tokenizer.word_index}
        word_list = [reverse_word_index[i + 1] for i in range(min(nb_words, len(reverse_word_index)))]
        dataset_info.feature_names = word_list
        if as_matrix:
            feature_matrix = tokenizer.texts_to_matrix(texts, mode=matrix_type)
            if getattr(dataset_info,'remove_stopwords', False):
                stopwords_list = get_stopwords_list()
                stopwords_feature_idxs = np.where(np.isin(word_list,list(stopwords_list)))
                feature_matrix = np.delete(feature_matrix,stopwords_feature_idxs,1)
                word_list = list(np.delete(np.array(word_list),stopwords_feature_idxs))
        else:
            feature_matrix = tokenizer.texts_to_sequences(texts)
    else:
        stopwords_list = None
        if getattr(dataset_info,'remove_stopwords', False):
            stopwords_list = list(get_stopwords_list())                
        if matrix_type == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, max_n), max_features=nb_words, stop_words = stopwords_list)
        else:
            vectorizer = CountVectorizer(ngram_range=(1, max_n), max_features=nb_words, stop_words = stopwords_list, binary=matrix_type == 'binary')
        feature_matrix = vectorizer.fit_transform(texts)   
        word_list = vectorizer.get_feature_names()   
    dataset_info.ds.df['features'] = list(feature_matrix.toarray())    
    dataset_info.feature_names = word_list

def get_mock_df(df_pk):
    df_pk.iloc[0]['folderName'] = 'MyWork' # Ensure at least 2 folder (label) values
    df_pk['label'] = df_pk['folderName']
    return df_pk


def get_pkl_features(pklFilePath, dataset_info, num_words=1000,matrix_type='binary',verbose=True,max_n=1):
    df_pk = pd.read_pickle(pklFilePath)
    print('Dataframe columns:\n--------------------------\n%s\n' % (list(df_pk.columns)))   
    df = df_pk # df = get_mock_df(df_pk)  # TODO:Debug:Remove - prepare mock df 
    labels = df['label'].unique().tolist()
    labelToNum = {labels[i]: i for i in range(len(labels))} 
    dct_labels_counts = dict(zip(list(df.groupby('label').groups.keys()),list(df.groupby('label')['label'].count())))
    print('Labels counts: %s ' % (dct_labels_counts))
    lst_lbl_counts =  list(dct_labels_counts.values())
    if  min(lst_lbl_counts) / max(lst_lbl_counts) < 0.7:
        print('***** Warning: Check for class imbalance')
    df['label_num'] = df.apply (lambda email: labelToNum[email.label], axis=1) 
    dataset_info.ds.df = df
    dataset_info.label_names = labels
    get_pkl_tokenzie_features(df, labels, dataset_info, num_words=num_words,matrix_type=matrix_type,verbose=verbose,max_n=max_n)
    if dataset_info.preprocess.modifyFeatureVector:
        dataset_info.ds.df = dataset_info.preprocess.modifyFeatureVector(dataset_info.ds.df,dataset_info)        
        
def get_pkl_tokenzie_features(df, labels,dataset_info, num_words=1000,matrix_type='binary',verbose=True,max_n=1):
    '''
    Read files exported by Spark and extract features 
    '''
    def concat_all_text(email):
        txt_all = ""
        for col_name in dataset_info.preprocess.text_cols:            
            txt_col = ""
            filtered_col_name = dataset_info.preprocess.filtered_prefix + col_name
            if dataset_info.preprocess.use_filtered and filtered_col_name in email:
                txt_col = email[filtered_col_name]
                col_name_used = filtered_col_name
            elif col_name in email:
                txt_col = email[col_name]
                col_name_used = col_name
            # Handle both str cols and tok_ cols (array of strings)
            if type(txt_col) == list:
                if len(txt_col) == 0:
                    txt_col = ''
                elif type(txt_col[0]) == str:
                    txt_col = ' '.join(txt_col)
                else:
                    raise Exception('Failed to concat column %s: It is of type list but not a string list' % (col_name_used))
            elif type(txt_col) == str or isinstance(txt_col, numbers.Number) or type(txt_col) == bool:
                txt_col = str(txt_col)
            else:
                raise Exception('Failed to concat column %s: type %s is not supported' % (col_name_used, type(txt_col)))
            txt_all += ' ' + txt_col
            
        return txt_all
    
    df['all_text'] = df.apply (concat_all_text, axis=1)
    texts = df['all_text'].tolist()    
    tokenize_vectorize(texts,labels, dataset_info,verbose, nb_words=num_words, as_matrix=True, matrix_type=matrix_type, max_n=max_n)
    
    
def get_ngram_data(emailsFilePath, dataset_info, num_words=1000,matrix_type='binary',verbose=True,max_n=1):
    cachefile = 'cache_data.tsv'  # Cached features csv file
    infofile = 'data_info.txt'
    emails = pd.read_csv(emailsFilePath, header=0, sep='\t', keep_default_na=False)
    dataset_info.ds.df = emails
    if dataset_info.force_papulate_cache or (not dataset_info.force_papulate_cache and not os.path.isfile(cachefile)):
        convert_emails(dataset_info.ds.df)
        write_csv(cachefile, dataset_info.ds.df, verbose=verbose)
        get_word_features(dataset_info, nb_words=num_words, matrix_type=matrix_type, verbose=verbose, max_n=max_n)
        write_info(infofile, dataset_info.label_names)
    else:
        cacheData = pd.read_csv(cachefile, sep='\t', header=0, keep_default_na=False)
        cachedEmail = cacheData[(cacheData.updateId.isin(emails.updateId))]
        noCachedEmail = emails[(~emails.updateId.isin(cacheData.updateId))]
        if not noCachedEmail.empty:
            raise Exception('Found emails not in cache')
        dataset_info.ds.df = cachedEmail
        get_word_features(dataset_info,nb_words=num_words, matrix_type=matrix_type, verbose=verbose, max_n=max_n)


def get_my_data(per_label=False):
    csvfile = 'my_data_%s.csv'%str(per_label)
    infofile = 'data_info.txt'
    if os.path.isfile(csvfile):
        features,labels,feature_names = read_csv(csvfile)
        label_names = read_info(infofile)
    else:
        mboxToBinaryCSV('.',csvfile,perLabel=per_label)
        features,labels,feature_names = read_csv(csvfile)#legacy code etc.
        label_names = list(set(labels))
        write_info(infofile,label_names)
    num_labels = max(labels)+1
    return features,labels,feature_names,label_names
        
def get_sequence_data():
    txtfile = 'sequence_data.txt'
    infofile = 'data_info.txt'
    if os.path.isfile(txtfile):
        features,labels = read_sequences(txtfile)
        label_names = read_info(infofile)
    else:
        emails = parseEmails('.')
        features,labels,words,labelVals = get_keras_features(emails,as_matrix=False)
        write_sequences(txtfile,features,labels)
        write_info(infofile,labelVals)
    num_labels = max(labels)+1
    return features,labels,label_names

def evaluate_mlp_model(dataset_info,num_classes,graph_to=None,verbose=True,extra_layers=0):
    print('\nClassifier: Keras evaluate_mlp_model\n')
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_classes)
    batch_size = 32
    nb_epoch = 7
    
    num_hidden=dataset_info.train.nn.num_hidden
    dropout=dataset_info.train.nn.dropout
            
    model = Sequential()
    model.add(Dense(num_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for i in range(extra_layers):
        model.add(Dense(num_hidden))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
    callbacks = []
    if graph_to is not None:
        plotter = Plotter(save_to_filepath=graph_to, show_plot_window=True)
        callbacks = [plotter]
    history = model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size, class_weight=get_class_weight(dataset_info),
                        verbose=1 if verbose else 0, validation_split=0.1,callbacks=callbacks)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1 if verbose else 0)
    if verbose:
        print('Test score:',score[0])
        print('Test accuracy (thresh=0.5): %f' % (score[1]))        

    class RetModel:
        def predict_proba(self,X):
            return model.predict(X)                
    return RetModel()

def evaluate_recurrent_model(dataset,num_classes):
    (X_train, Y_train), (X_test, Y_test) = dataset
    max_features = 20000
    maxlen = 125  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    print("Pad sequences (samples x time) with maxlen %d"%maxlen)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(GRU(512))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer='adam')

    print("Train...")
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=15,
              validation_data=(X_test, Y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, Y_test,
                                batch_size=batch_size,
                                show_accuracy=True)
    if verbose:
        print('Test score:', score)
        print('Test accuracy:', acc)
    return score[1]

def evaluate_conv_model(dataset, num_classes, maxlen=125,embedding_dims=250,max_features=5000,nb_filter=300,filter_length=3,num_hidden=250,dropout=0.25,verbose=True,pool_length=2,with_lstm=False):
    (X_train, Y_train), (X_test, Y_test) = dataset
    
    batch_size = 32
    nb_epoch = 7

    if verbose:
        print('Loading data...')
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print('Pad sequences (samples x time)')
    
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    if verbose:
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        print('Build model...')

    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(dropout))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Conv1D(activation="relu", filters=nb_filter, kernel_size=filter_length, strides=1, padding="valid"))
    if pool_length:
        # we use standard max pooling (halving the output of the previous layer):
        model.add(MaxPooling1D(pool_size=2))
    if with_lstm:
        model.add(LSTM(125))
    else:
        # We flatten the output of the conv layer,
        # so that we can add a vanilla dense layer:
        model.add(Flatten())

        #We add a vanilla hidden layer:
        model.add(Dense(num_hidden))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size,epochs=nb_epoch, validation_split=0.1)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1 if verbose else 0)
    if verbose:
        print('Test score:',score[0])
        print('Test accuracy:', score[1])
    predictions = model.predict_classes(X_test,verbose=1 if verbose else 0)
    return predictions,score[1]
