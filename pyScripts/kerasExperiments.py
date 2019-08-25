from kerasClassify import make_dataset, get_ngram_data, evaluate_mlp_model, get_emails, write_csv, get_pkl_features
from mlClassify import logreg_classifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from keras.utils import np_utils

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd
from hpyutils import MyObj, setattrs
from my_metrics import calc_metrics, print_metrics, plot_metrics
from debug_ml import explain_predictions



def init_config():
    
    dataset_info = MyObj()
    
    dataset_info.csvEmailsFilePath =  "./data/enron_6_email_folders_Inboxes_KAMINSKI.tsv"; # Dataset tsv file path. Each line is an email
    dataset_info.num_runs = 1
    # PreProcessing
    dataset_info.remove_stopwords = True # remove stopwords (english only for now)
    dataset_info.ngram_max = 2 # Max number of word ngrams (1 for unigram, 2 for bigram)
    dataset_info.vocab_size = 10000
    dataset_info.feature_type = 'tfidf' # Type of feature in matrix: binary (0/1), tfidf, count
    dataset_info.use_keras_tokenizer = False
    # Features
    dataset_info.toccDomains = True # Use to and cc email domains as features 
    #-- Data 
    # dataset_info.new_label_names = ['Save','DontSave'] # random select labels to map to one of the labels in array. mutually ex with labels_map
    dataset_info.labels_map = { 'Inbox' : 'DontSave','Notes inbox' : 'DontSave', 'default_mapping' : 'Save' } # manual mapping with default mapping
    #dataset_info.labels_map = { 'Inbox' : 'DontSave','Notes inbox' : 'DontSave', 'default_mapping' : 'Omit', 'Projects': 'Save'} # manual mapping with default mapping
    dataset_info.labels_map_filter_names = ['Omit']# values of labels to filter out in df
    dataset_info.sub_sample_mapped_labels = { 'Save': 650 ,'DontSave' : 650 }
    #dataset_info.class_weight = { 'Save': 6 ,'DontSave' : 1 }
    # dataset_info.new_total_samples = 100
    dataset_info.test_split = 0.1
    
    #save final dataframe to csv file only in case num_runs=1
    dataset_info.save_df = False
    
    #--force papulate cache
    dataset_info.force_papulate_cache = False
    
    
    dataset_info.random_seed = []
    
    ######### Preprocessing ###########
    dataset_info.preprocess = MyObj()
    setattrs(dataset_info.preprocess,
         text_cols = [ 'subject', 'content', 'to','cc'], # , 'people_format' # Important: Not used in old get_ngrams_data (.tsv)
         modifyFeatureVector = None, # accept (dataset_info.ds.df,dataset_info) and return a new df, modifying df['feature'] using Embeddings, KB 
         select_best = 4000, # Number of features to keep in feature selection (disable if working )
         use_filtered = True,
         filtered_prefix = 'filt_',     
    )
               
    ########################################### Training #####################################################
    dataset_info.train = MyObj()
    setattrs(dataset_info.train,
        classifier_func = logreg_classifier # evaluate_mlp_model # Default is Keras:
    )
    
    #-- NN Arch
    dataset_info.train.nn = MyObj()
    setattrs(dataset_info.train.nn,
        num_hidden = 512,
        dropout = 0.5,
    )
    
    ########################################### Metrics #####################################################
    dataset_info.metrics = MyObj()
    setattrs(dataset_info.metrics,
      fpr_thresh = 0.1, # Requires max fpr of 0.1 --> calc class proba threshold for binary classification 
      report_metrics=['sel_tpr','sel_fpr','roc_auc', 'accuracy','precision','recall','f_score'], # Specify metrics from new_metrics to report (see metrics names in my_metrics.py)    
      # testgroupby = 'sender' or 'to' # Report accuracy (correct predictions), by number of training samples per group (groupby sender test samples, for each group get its training samples)
    )
    
    ########################################### Hooks #####################################################
    dataset_info.hooks = MyObj()
    setattrs(dataset_info.hooks,
       afterFeatures = None, # function that is called after tokenization + feature extraction, before make_dataset (train/test split and subsample)
    )
    
    ######################## End Enron derived datasets experiments ##########################################
    if dataset_info.num_runs > 1 and dataset_info.save_df:
        raise Exception("Cannot use both save_df and num_runs > 1")
    
    ### Experiment params validation and computed params
    if getattr(dataset_info,'labels_map',False) :
        if getattr(dataset_info, 'new_label_names', False):
            raise Exception("Cannot use both new_label_names and labels_map")
        # Create new_label_names form labels_map unique values
        dataset_info.new_label_names = [i for i in list(set(dataset_info.labels_map.values())) if i not in getattr(dataset_info, "labels_map_filter_names", [])]
    
    return dataset_info

dataset_info = init_config()

class Dataset():
    def __init__(self):
        pass

    # sorted - get df sorted by column "index_row"
    def get_df(self, sorted=True):
        df = self.df
        if hasattr(df, "label_filter_out"):
            df = df[~df[getattr(self, "filer_col_name", "label_filter_out")] == True]
        if sorted and hasattr(self.df, 'index_row'):
            df = df.sort_values(by=['index_row'])
        return df

    def get_X_train(self):
        return self.df[(~self.df[getattr(self, 'train_col_name', 'train')].isnull()) &
                       (~self.df[getattr(self, "filer_col_name", "label_filter_out")] == True)].sort_values(by=['index_row'])

    def get_X_test(self):
        return self.df[(~self.df['test'].isnull()) & (~self.df[getattr(self, "filer_col_name", "label_filter_out")] == True)].sort_values(by=['index_row'])

    def get_Y_train(self, X_train, to_categorical=False, num_labels=0):
        Y_train = X_train[getattr(self, 'label_col_name', 'label_num')].tolist()
        if (to_categorical):
            return np_utils.to_categorical(Y_train, num_labels)
        return Y_train

    def get_Y_test(self, X_test, to_categorical=False, num_labels=0):
        Y_test = X_test[getattr(self, 'label_col_name', 'label_num')].tolist()
        if (to_categorical):
            return np_utils.to_categorical(Y_test, num_labels)
        return Y_test

    def get_dataset(self, to_categorical=False, num_labels=0):
        X_train = self.get_X_train()
        X_test = self.get_X_test()
        Y_train = self.get_Y_train(X_train, to_categorical=to_categorical, num_labels=num_labels)
        Y_test = self.get_Y_test(X_test, to_categorical=to_categorical, num_labels=num_labels)
        X_train_features = np.array(X_train['features'].tolist())        
        X_test_features = np.array(X_test['features'].tolist())
        if hasattr(self, 'selected_features_idxs'):
            X_train_features = X_train_features[:, self.selected_features_idxs]
            X_test_features = X_test_features[:, self.selected_features_idxs]
        return (X_train_features, Y_train), (X_test_features, Y_test)
    
    def setFilterCol(self,new_filter_col_name, logicOr = True):
        '''
        Sets a new filter column
        new_filter_col_name - col name of existing boolean column (filter according to new condition)
        logicOr - logic OR the new col with old filter col (if old exist), to combine old and new filters. if False - only new filter is in effect
        '''
        old_filter_col_name = getattr(self,'filer_col_name',None)
        if old_filter_col_name and logicOr:
            self.df[new_filter_col_name] = self.df[new_filter_col_name] | self.df[old_filter_col_name]        
        
        dataset_info.ds.filer_col_name = new_filter_col_name        
        

def select_best_features(dataset_info, num_labels, num_best, verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    if verbose:
        print('\nSelecting %d best features\n'%num_best)
    selector = SelectKBest(chi2, k=num_best)
    selector.fit_transform(X_train, Y_train)
    dataset_info.ds.selected_features_idxs = selector.get_support(indices=True).tolist()

    return selector.scores_

def plot_feature_scores(feature_names,scores,limit_to=None,save_to=None,best=True):
    plt.figure()
    if best:
        plt.title("Best features")
    else:
        plt.title("Worst features")
    if limit_to is None:
        limit_to = len(features_names)
    #for some reason index 0 always wrong
    scores = np.nan_to_num(scores)
    if best:
        indices = np.argsort(scores)[-limit_to:][::-1]
    else:
        indices = np.argsort(scores)[:limit_to]
    #indices = np.argpartition(scores,-limit_to)[-limit_to:]
    plt.bar(range(limit_to), scores[indices],color="r", align="center")
    plt.xticks(range(limit_to),np.array(feature_names)[indices],rotation='vertical')
    plt.xlim([-1, limit_to])
    plt.ylabel('Score')
    plt.xlabel('Word')
    plt.show(block=False)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')


def make_plot(x,y,title=None,x_name=None,y_name=None,save_to=None,color='b',new_fig=True):
    if new_fig:
        plt.figure()
    plot = plt.plot(x,y,color)
    if title is not None:
        plt.title(title)
    if x_name is not None:
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')
    return plot

def make_plots(xs,ys,labels,title=None,x_name=None,y_name=None,y_bounds=None,save_to=None):
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    handles = []
    plt.figure()
    plt.hold(True)
    for i in range(len(labels)):
        plot, = make_plot(xs[i],ys[i],color=colors[i%len(colors)],new_fig=False)
        handles.append(plot)
    plt.legend(handles,labels)
    if title is not None:
        plt.title(title)
    if x_name is not None:
        plt.xlabel(x_name)
    if y_name is not None:
        plt.ylabel(y_name)
    if y_bounds is not None:
        plt.ylim(y_bounds)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight')
    plt.hold(False)

def get_baseline_dummy(dataset_info, num_labels,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    dummy = DummyClassifier()
    dummy.fit(X_train,dataset_info.ds.get_Y_train(X_train))
    predictions = dummy.predict(X_test)
    accuracy = accuracy_score(dataset_info.ds.get_Y_test(X_test),predictions)
    
    if verbose:
        print('Got baseline of %f with dummy classifier'%accuracy)

    return accuracy

def get_baseline_svm(dataset_info,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    linear = LinearSVC(penalty='l1',dual=False)
    grid_linear = GridSearchCV(linear, {'C':[0.1, 0.5, 1, 5, 10]}, cv=5)
    grid_linear.fit(X_train,dataset_info.ds.get_Y_train(X_train))
    accuracy = grid_linear.score(X_test, dataset_info.ds.get_Y_test(X_train))
    
    if verbose:
        print('Got baseline of %f with svm classifier'%accuracy)

    return accuracy

def get_baseline_knn(dataset_info,num_labels,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    knn = KNeighborsClassifier(n_neighbors=100,n_jobs=-1)
    knn.fit(X_train,dataset_info.ds.get_Y_train(X_train))
    predictions = np.round(knn.predict(X_test))
    accuracy = accuracy_score(dataset_info.ds.get_Y_test(X_train),predictions)

    if verbose:
        print('Got baseline of %f with linear regression '%accuracy)

    return accuracy

def get_baseline_pa(dataset_info,verbose=True):
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    classifier = PassiveAggressiveClassifier(n_jobs=-1,fit_intercept=True)
    classifier.fit(X_train,dataset_info.ds.get_Y_train(X_train))
    accuracy = classifier.score(X_test,dataset_info.ds.get_Y_test(X_train))
    
    if verbose:
        print('Got baseline of %f with Passive Aggressive classifier'%accuracy)

    return accuracy
     
def run_once(verbose=True,test_split=0.1,ftype='binary',num_words=10000,select_best=4000, plot=True,plot_prefix='',graph_to=None):    
    # Prepare features
    dataset_info.ds = Dataset()
    if getattr(dataset_info,'read_exp_pkl',None):
        get_pkl_features(dataset_info.csvEmailsFilePath,dataset_info, num_words=num_words,matrix_type=ftype,verbose=verbose, max_n=dataset_info.ngram_max)
    else:
        get_ngram_data(dataset_info.csvEmailsFilePath,dataset_info, num_words=num_words,matrix_type=ftype,verbose=verbose, max_n=dataset_info.ngram_max)

    # User defined hook to optionally modify data before train/test split
    if dataset_info.hooks.afterFeatures:
        dataset_info.hooks.afterFeatures(dataset_info)
        
    num_labels = len(dataset_info.label_names)    
    # Create dataset including splits, sub sampling, labels mapping
    # ((X_train,Y_train_c),(X_test,Y_test_c)),Y_train,Y_test,num_labels
    num_labels = make_dataset(dataset_info)    
    if select_best and select_best<num_words:
        scores = select_best_features(dataset_info,num_labels,select_best,verbose=verbose)
    if plot and select_best:
        feature_names = dataset_info.feature_names
        plot_feature_scores(feature_names, scores,limit_to=25, save_to=plot_prefix+'scores_best.png')
        plot_feature_scores(feature_names, scores,limit_to=25, save_to=plot_prefix+'scores_worst.png',best=False)
    
    # Print train/test stats 
    (X_train, Y_train), (X_test, Y_test) = dataset_info.ds.get_dataset(to_categorical=True, num_labels=num_labels)
    if verbose:
       print(X_train.shape[0], 'train sequences')
       print(X_test.shape[0], 'test sequences')
       print('X_train shape:', X_train.shape)
       print('X_test shape:', X_test.shape)
       print('Y_train shape:', Y_train.shape)
       print('Y_test shape:', Y_test.shape)
       print('Building model...')
    
    # Train a model    
    model = dataset_info.train.classifier_func(dataset_info,num_labels,graph_to=graph_to, verbose=verbose) 
    
    # Evaluate: ROC, confusion matrix, plots
    new_metrics,predictions = calc_metrics(num_labels,model,dataset_info)
    
    label_names = dataset_info.label_names
    # Verbose print and plot
    if getattr(dataset_info,'new_label_names',None):
        label_names = dataset_info.new_label_names
    if verbose:        
        print_metrics(new_metrics)
    if plot:
        plot_metrics(new_metrics,label_names)
        # Explain important features
        # explain_predictions(dataset,predictions,model,feature_names,label_names)
        
    return num_labels,new_metrics

def test_features_words():
    #get emails once to pickle
    emails = get_emails(dataset_info.csvEmailsFilePath,verbose=False)

    types = ['binary','count','freq','tfidf']
    all_accs = []
    all_counts = []
    all_times = []
    all_baselines = []
    maxacc = 0
    maxtype = None
    for ftype in types:
        word_counts = range(500,3600,250)
        all_counts.append(word_counts)
        accs=[]
        times=[]
        baselines = []
        print('\nTesting learning for type %s with word counts %s\n'%(ftype,str(word_counts)))
        for word_count in word_counts:
            start = time.time()
            num_labels,acc_one = run_once(num_words=word_count,ftype=ftype,plot=False,verbose=False,select_best=None)
            acc = (acc_one+sum([run_once(num_words=word_count,ftype=ftype,plot=False,verbose=False,select_best=None)[3] for i in range(4)]))/5.0
            end = time.time()
            elapsed = (end-start)/5.0
            times.append(elapsed)
            print('\nGot acc %f for word count %d in %d seconds'%(acc,word_count,elapsed))

            start = time.time()
            baseline = get_baseline_dummy(dataset_info, num_labels,verbose=False)
            baselines.append(baseline)
            end = time.time()
            belapsed = end-start
            print('Got baseline acc %f in %d seconds'%(baseline,belapsed))

            if acc>maxacc:
                maxacc = acc
                maxtype = ftype
            accs.append(acc)
        all_baselines.append(baselines)
        all_times.append(times)
        all_accs.append(accs)
        print('\nWord count accuracies:%s\n'%str(accs))
    make_plots(all_counts,all_accs,types,title='Test accuracy vs max words',y_name='Test accuracy',x_name='Max most frequent words',save_to='word_accs.png',y_bounds=(0,1))
    make_plots(all_counts,all_accs,types,title='Test accuracy vs max words',y_name='Test accuracy',x_name='Max most frequent words',save_to='word_accs_zoomed.png',y_bounds=(0.6,0.95))
    make_plots(all_counts,all_baselines,types,title='Baseline accuracy vs max words',y_name='Baseline accuracy',x_name='Max most frequent words',save_to='word_baseline_accs.png',y_bounds=(0,1))
    make_plots(all_counts,all_times,types,title='Time vs max words',y_name='Parse+test+train time (seconds)',x_name='Max most frequent words',save_to='word_times.png')
    print('\nBest word accuracy %f with features %s\n'%(maxacc,maxtype))

def test_hidden_dropout():
    #get emails once to pickle
    emails = get_emails(dataset_info.csvEmailsFilePath,verbose=False)

    dropouts = [0.25,0.5,0.75]
    all_accs = []
    all_counts = []
    all_times = []
    maxacc = 0
    maxh = 0
    for d in dropouts:
        hidden = [32,64,128,256,512,1024,2048]
        all_counts.append(hidden)
        accs=[]
        times=[]
        print('\nTesting learning for dropout %f with hidden counts %s\n'%(d,str(hidden)))
        for h in hidden:
            start = time.time()
            acc = sum([run_once(dropout=d,num_words=2500,num_hidden=h,plot=False,verbose=False,select_best=None)[3] for i in range(5)])/5.0
            end = time.time()
            elapsed = (end-start)/5.0
            times.append(elapsed)
            print('\nGot acc %f for hidden count %d in %d seconds'%(acc,h,elapsed))
            if acc>maxacc:
                maxacc = acc
                maxh = h
            accs.append(acc)
        all_times.append(times)
        all_accs.append(accs)
        print('\nWord count accuracies:%s\n'%str(accs))
    make_plots(all_counts,all_accs,['Droupout=%f'%d for d in dropouts],title='Test accuracy vs num hidden',y_name='Test accuracy',x_name='Number of hidden units',save_to='hidden_accs.png',y_bounds=(0,1))
    make_plots(all_counts,all_accs,['Droupout=%f'%d for d in dropouts],title='Test accuracy vs num hidden',y_name='Test accuracy',x_name='Number of hidden units',save_to='hidden_accs_zoomed.png',y_bounds=(0.8,1))
    make_plots(all_counts,all_times,['Droupout=%f'%d for d in dropouts],title='Time vs max words',y_name='Parse+test+train time (seconds)',x_name='Number of hidden units',save_to='hidden_times.png')
    print('\nBest word accuracy %f with hidden %d\n'%(maxacc,maxh))

def test_select_words(num_hidden=512):
    #get emails once to pickle
    emails = get_emails(dataset_info.csvEmailsFilePath,verbose=False)

    word_counts = [2500,3500,4500,5500]
    all_accs = []
    all_counts = []
    all_times = []
    maxacc = 0
    maxs = None
    for word_count in word_counts:
        select = [0.5,0.6,0.7,0.8,0.9]
        all_counts.append(select)
        accs=[]
        times=[]
        print('\nTesting learning for word count %d with selects %s\n'%(word_count,str(select)))
        for s in select:
            start = time.time()
            acc = sum([run_once(num_hidden=num_hidden,dropout=0.1,num_words=word_count,plot=False,verbose=False,select_best=int(s*word_count))[3] for i in range(5)])/5.0
            end = time.time()
            elapsed = (end-start)/5.0
            times.append(elapsed)
            print('\nGot acc %f for select ratio %f in %d seconds'%(acc,s,elapsed))
            if acc>maxacc:
                maxacc = acc
                maxs = s
            accs.append(acc)
        all_times.append(times)
        all_accs.append(accs)
        print('\nWord count accuracies:%s\n'%str(accs))
    make_plots(all_counts,all_accs,['Words=%d'%w for w in word_counts],title='Test accuracy vs ratio of words kept',y_name='Test accuracy',x_name='Ratio of best words kept',save_to='select_accs_%d.png'%num_hidden,y_bounds=(0,1))
    make_plots(all_counts,all_accs,['Words=%d'%w for w in word_counts],title='Test accuracy vs ratio of words kept',y_name='Test accuracy',x_name='Ratio of best words kept',save_to='select_accs_zoomed_%d.png'%num_hidden,y_bounds=(0.8,1))
    make_plots(all_counts,all_times,['Words=%d'%w for w in word_counts],title='Time vs ratio of words kept',y_name='Parse+test+train time (seconds)',x_name='Ratio of best words kept',save_to='select_times_%d.png'%num_hidden,y_bounds=(0,65))
    print('\nBest word accuracy %f with select %f\n'%(maxacc,maxs))


# ------- Experiments -----------------------------
# True to run feature extraction, selection + svm baseline (~ 0.78)
    
run_baseline = False
#test_features_words()
#test_hidden_dropout()
#test_select_words(128)
#test_select_words(32)
#test_select_words(16)

# TODO: try ftype = 'tfidf'

def output_run_stats(df_test_metrics):        
    print('Test single runs stats:')
    print(df_test_metrics)


def run_exp():
    '''
    Multiple runs of a single config (for avg stats)
    '''
    # Create metrics tracking dataframe for multiple runs, where each column is a metric (acc,prec,recall,f1 ...)
    import io
    from datetime import datetime
    start_time = datetime.now()
    metrics_columns=[mtr_name for mtr_name in dataset_info.metrics.report_metrics]
    metrics_dtype=[np.float for d in range(0,len(metrics_columns))]
    df_test_metrics = pd.read_csv(io.StringIO(""), names=metrics_columns, dtype=dict(zip(metrics_columns,metrics_dtype))) # pd.DataFrame(columns=metrics_columns,dtype=metrics_dtype)
    dataset_info.state = MyObj()
    for i in range(0,dataset_info.num_runs):
        if len(dataset_info.random_seed) <= i:
            dataset_info.random_seed.append(int(time.time()))
        dataset_info.state.index_random_seed = i
        *dummy,new_metrics = run_once(num_words=dataset_info.vocab_size,ftype=dataset_info.feature_type,test_split=dataset_info.test_split, plot=False if dataset_info.num_runs > 1 else True, verbose=True,select_best=dataset_info.preprocess.select_best)
        df_test_metrics.loc[i] = [getattr(new_metrics,mtr_name) for mtr_name in dataset_info.metrics.report_metrics]
                
    print('random seed {}:', dataset_info.random_seed)
    output_run_stats(df_test_metrics)
    
    if hasattr(dataset_info, 'save_df') and dataset_info.save_df:
        write_csv('final_df.tsv', dataset_info.ds.df, verbose=True)
    
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    return df_test_metrics


########################## Running multipe configs (datset_info), each num_runs ########################################
def run_multi_exps_configs(exps):
    global dataset_info
    df_results = pd.DataFrame() # array of df - a df per config
    for exp in exps:
        dataset_info = exp.dataset_info
        df_test_metrics = run_exp()
        exp.tag_metrics(dataset_info,df_test_metrics)
        df_results = pd.concat([df_results,df_test_metrics])
    return df_results

class BaseExp:
    def __init__(self):
       self.dataset_info = init_config()
    def tag_metrics(self,dataset_info,df_test_metrics):
       pass
        