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
from mboxConvert import parseEmails,parseEmailCSV, parseEmailsCSV,getEmailStats,mboxToBinaryCSV
from kerasPlotter import Plotter
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM,GRU
from keras.layers.core import Dense, Dropout, Activation, Flatten

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

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

def get_word_features(emails, dataset_info,verbose=True, nb_words=5000, skip_top=0, maxlen=None, as_matrix=True, matrix_type='count',
                      label_cutoff=0.01, max_n=1):
    (totalWordsCount, fromCount, domainCount, labels) = getEmailStats(emails)
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
    for index, email in emails.iterrows():
        labelCounts[email.label] += 1
    cutoff = int(len(emails) * label_cutoff)
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
    for index, email in emails.iterrows():
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
    if max_n == 1 or not as_matrix:
        tokenizer = Tokenizer(nb_words)
        tokenizer.fit_on_texts(texts)
        reverse_word_index = {tokenizer.word_index[word]: word for word in tokenizer.word_index}
        word_list = [reverse_word_index[i + 1] for i in range(min(nb_words, len(reverse_word_index)))]
        if as_matrix:
            feature_matrix = tokenizer.texts_to_matrix(texts, mode=matrix_type)
            if getattr(dataset_info,'remove_stopwords', False):
                stopwords_list = get_stopwords_list()
                stopwords_feature_idxs = np.where(np.isin(word_list,list(stopwords_list)))
                feature_matrix = np.delete(feature_matrix,stopwords_feature_idxs,1)
                word_list = list(np.delete(np.array(word_list),stopwords_feature_idxs))
            return updateIds, feature_matrix, emailLabels, word_list, labels
        else:
            sequences = tokenizer.texts_to_sequences(texts)
            return updateIds, sequences, emailLabels, word_list, labels
    else:
        if matrix_type == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=(1, max_n), max_features=nb_words)
        else:
            vectorizer = CounterVectorizer(ngram_range=(1, max_n), max_features=nb_words,
                                           binary=matrix_type == 'binary')
        feature_matrix = vectorizer.fit_transform(texts)
        word_list = vectorizer.get_feature_names()
        return updateIds, feature_matrix, emailLabels, word_list, labels

def write_csv(csvfile, emails, verbose=True):
    emails.to_csv(csvfile, index=False)
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
    if hasattr(dataset_info, 'random_seed') and not dataset_info.random_seed is None:
        np.random.seed(dataset_info.random_seed)

def get_idx_to_new_label_dict(dataset_info):
  '''
  Return a dict that maps label idx to label name 0-->Save 1 --> DontSave ...
  '''
  return dict(zip(list(range(0,len(dataset_info.new_label_names))), dataset_info.new_label_names))             

def auto_subsample_dataset(X,labels, dataset_info):
    '''
    Reduce dataset to dataset_info.new_total_samples, taking the same number of samples from every class.
    '''
    if not hasattr(dataset_info, 'new_total_samples') or dataset_info.new_total_samples is None or dataset_info.new_total_samples == 0:
        return (X,labels)
    unq_lbls = np.unique(labels, return_counts=True)
    n_samples_per_label = dataset_info.new_total_samples * (1 - dataset_info.test_split) / len(unq_lbls[0])
    if any(unq_lbls[1] < min(15,n_samples_per_label) ):
        raise Exception("One of the classes doesn't have enough labels %d, unq_lbls[1]=%s" % (n_samples_per_label, ','.join(unq_lbls[1])) )
    per_label_cnt = Counter()
    new_X = []
    new_labels = []
    # Accumulate reduced dataset (trainset) in new_X while the number of samples from each label isn't higher than n_samples_per_label
    for i in range(0,len(X)):
        if (per_label_cnt[labels[i]] < n_samples_per_label):
            new_X.append(X[i])
            new_labels.append(labels[i])
            per_label_cnt[labels[i]] += 1
    return (np.array(new_X),new_labels)

def subsample_dataset_by_label_stratified(X,labels, dataset_info):
    '''
    Reduce each label in dataset to specified amount in dataset_info.sub_sample_mapped_labels ex: { 'Save': 60 ,'DontSave' : 600 }
    '''
    if not hasattr(dataset_info, 'sub_sample_mapped_labels') or dataset_info.sub_sample_mapped_labels is None or len(dataset_info.sub_sample_mapped_labels) == 0:
        return (X,labels)
    
    unq_lbls = np.unique(labels)
    init_randomness(dataset_info)
    
    # find observation index of each class levels
    groupby_lbls = {}
    for ii, lbl_idx in enumerate(unq_lbls):
        lbl_samples_idxs = [idx for idx, val in enumerate(labels) if val == lbl_idx]
        groupby_lbls[lbl_idx] = lbl_samples_idxs 
    
    # Undersample each label (in a loop) according to sub_sample_mapped_labels (above)
    idx_to_new_label = get_idx_to_new_label_dict(dataset_info)     
    under_sample_idxs = [] # Holds all idx of samples (from all labels), selected to keep
    for lbl_idx, lbl_samples_idxs in groupby_lbls.items():
        lbl_sample_size = dataset_info.sub_sample_mapped_labels[idx_to_new_label[lbl_idx]]
        lbl_under_sample_idxs = np.random.choice(lbl_samples_idxs, size=lbl_sample_size, replace=False).tolist()
        under_sample_idxs += lbl_under_sample_idxs
    return (X[under_sample_idxs],[labels[sample_idx] for sample_idx in under_sample_idxs])

def get_class_weight(dataset_info):
    if not hasattr(dataset_info, 'class_weight') or dataset_info.class_weight is None:
        return None
    
    new_label_to_idx = dict(zip(dataset_info.new_label_names, list(range(0,len(dataset_info.new_label_names)))))
    return { new_label_to_idx[lbl_name]:dataset_info.class_weight[lbl_name] for lbl_name in dataset_info.class_weight.keys()}

def create_get_new_label_idx(dataset_info, new_total_labels):  
    permuted_old_label_idxs = np.random.permutation(len(dataset_info.label_names))
    dataset_info.permuted_old_label_idxs = permuted_old_label_idxs
    def get_new_label_idx(old_label_idx):
        permuted_old_label_idx = permuted_old_label_idxs[old_label_idx]
        return int(permuted_old_label_idx % new_total_labels)
    return get_new_label_idx

    
def auto_map_labels(Y_train,Y_test, dataset_info):
    '''
    auto map the new_total_labels - permute labels and map each group to a new label 
    '''
    new_total_labels = len(dataset_info.new_label_names)
    old_num_labels = len(dataset_info.label_names)    
    if not (old_num_labels / new_total_labels).is_integer():
        raise Exception("old_num_labels must be integer %f" % (old_num_labels))
        
    get_new_label_idx = create_get_new_label_idx(dataset_info, new_total_labels)    
    new_Y_train = list(map(get_new_label_idx, Y_train))
    new_Y_test = list(map(get_new_label_idx, Y_test))        
    return (new_Y_train, new_Y_test, new_total_labels)        

def map_labels(Y_train,Y_test, dataset_info):
    '''
    Manually map labels according to user supplied labels_map
    '''
    # First map orig label idx (3) --> orig label name (ex: 'Inbox') --> new label name ('Save') --> new label idx
    orig_label_idx_to_name = dict(zip(list(range(0,len(dataset_info.label_names))),dataset_info.label_names))     
    new_label_to_idx = dict(zip(dataset_info.new_label_names, list(range(0,len(dataset_info.label_names)))))     
    def get_new_label_idx(idx_orig_label):
        orig_label = orig_label_idx_to_name[idx_orig_label] # 0 --> 'Inbox'
        # If not found in labels_map, try to use 'all_others' key if exist
        if orig_label in dataset_info.labels_map:
            new_label = dataset_info.labels_map[orig_label] # { 'Inbox' : 'Save','Notes inbox' : 'Save', 'default_mapping' : 'DontSave' }
        elif 'default_mapping' in dataset_info.labels_map:
            new_label = dataset_info.labels_map['default_mapping']
        else:
            raise Exception("Failed to map %s"%(orig_label))
        
        new_label_idx = new_label_to_idx[new_label]    
        return new_label_idx
    
    new_Y_train = list(map(get_new_label_idx, Y_train))
    new_Y_test = list(map(get_new_label_idx, Y_test))        
    new_total_labels = len(dataset_info.new_label_names)
    return (new_Y_train, new_Y_test, new_total_labels)   

def make_dataset(features,labels,dataset_info,test_split=0.1,nb_words=1000):
    '''
    Split train, test subsample and remap labels
    '''
    num_labels = len(dataset_info.label_names)   
    num_examples = features.shape[0]    
    random_order = np.random.permutation(num_examples)
    index_split = (int)(test_split*num_examples)
    train_indices = random_order[index_split:]
    test_indices = random_order[:index_split]
    X_train = features[train_indices]
    X_test = features[test_indices]
    Y_train = [labels[i] for i in train_indices]
    Y_test = [labels[i] for i in test_indices]    
    # Subsample dataset to get dataset_info.new_total_samples 
    (X_train,Y_train) = auto_subsample_dataset(X_train,Y_train, dataset_info)
    # Map labels (ex: from folders to binary 2 folder groups)
    if dataset_info.new_label_names and not hasattr(dataset_info,'labels_map'):
        (Y_train,Y_test, num_labels) = auto_map_labels(Y_train,Y_test, dataset_info)
    # Map labels manually    
    if dataset_info.labels_map:
        (Y_train,Y_test, num_labels) = map_labels(Y_train,Y_test, dataset_info)
    # Subsample manually according to [optional] sub_sample_mapped_labels. 
    # Note that train/test split (0.1) already occured (above) --> so only trainset is reduced 
    (X_train,Y_train) = subsample_dataset_by_label_stratified(X_train,Y_train, dataset_info)
    Y_train_c = np_utils.to_categorical(Y_train, num_labels)
    Y_test_c = np_utils.to_categorical(Y_test, num_labels)
    return ((X_train,Y_train_c),(X_test,Y_test_c)),Y_train,Y_test,num_labels

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

def convert_emails(emails):
    parsedEmails = emails.apply(lambda email: parseEmailCSV(email), axis=1)
    dataframe_emails = pd.DataFrame.from_records(parsedEmails,
                              columns=["updateId", "label", "subject", "sender", "fromDomain", "to","toDomain", "cc","ccDomain", "day",
                                       "date", "month", "year", "hour", "content", "words"])
    return dataframe_emails

def get_ngram_data(emailsFilePath, dataset_info, num_words=1000,matrix_type='binary',verbose=True,max_n=1):
    cachefile = 'cache_data.csv'  # Cached features csv file
    infofile = 'data_info.txt'
    emails = pd.read_csv(emailsFilePath, header=0, sep='\t', keep_default_na=False)
    if dataset_info.force_papulate_cache or (not dataset_info.force_papulate_cache and not os.path.isfile(cachefile)):
        parsedEmails = convert_emails(emails)
        updateIds, features, labels, feature_names, label_names = get_word_features(parsedEmails,dataset_info,
                                                                                    nb_words=num_words,
                                                                                    matrix_type=matrix_type,
                                                                                    verbose=verbose, max_n=max_n)
        if max_n == 1:
            write_csv(cachefile, parsedEmails, verbose=verbose)
            write_info(infofile, label_names)
    else:
        cacheData = pd.read_csv(cachefile, header=0, keep_default_na=False)
        cachedEmail = cacheData[(cacheData.updateId.isin(emails.updateId))]
        noCachedEmail = emails[(~emails.updateId.isin(cacheData.updateId))]
        if not noCachedEmail.empty:
            raise Exception('Found emails not in cache')
        updateIds, features, labels, feature_names, label_names = get_word_features(cachedEmail, dataset_info,nb_words=num_words, matrix_type=matrix_type, verbose=verbose, max_n=max_n)
    return features, labels, feature_names, label_names

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

def evaluate_mlp_model(dataset,dataset_info,num_classes,extra_layers=0,num_hidden=512,dropout=0.5,graph_to=None,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset
    batch_size = 32
    nb_epoch = 7
            
    if verbose:
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        print('Y_train shape:', Y_train.shape)
        print('Y_test shape:', Y_test.shape)
        print('Building model...')
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
                
    return score,model

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
