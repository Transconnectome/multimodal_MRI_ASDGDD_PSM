from collections import defaultdict

import numpy as np
import pandas as pd
from psmpy import PsmPy
from psmpy.functions import cohenD
from sklearn.preprocessing import OneHotEncoder 
from skmultilearn.model_selection import IterativeStratification

from config import Config
import pdb
def iter_strat(labels, n_splits, args):
    binary_cols = list(set(Config.iter_strat_label['binary'] + args.cat_target)) #add the label_name if it is new
    floatized_arr = multilabel_matrix_maker(labels, n_chunks=8,
                                            binary_cols=binary_cols, 
                                            multiclass_cols=Config.iter_strat_label['multiclass'],
                                            continuous_cols=Config.iter_strat_label['continuous'])

    skf_target = [floatized_arr, floatized_arr]        
    kf = IterativeStratification(n_splits=n_splits, order=15, random_state = np.random.seed(args.split_seed))
    return [*kf.split(*skf_target)]


def psm_iterative_stratification(imageFiles_labels, labels_total, args):
    def run_psm(labels, treatment, args):
        df_psm = labels.copy().loc[:,[args.subjectkey, treatment]+Config.psm_label]
        psm = PsmPy(df_psm, treatment=treatment, indx=args.subjectkey)
        psm.logistic_ps(balance=False)
        psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=1)
        psm.effect_size_plot()
        print(f"{'effect size':=^30}")
        print(psm.effect_size)
        if 'save_psm' in args.exp_name:
            psm.df_matched.to_csv('/scratch/x2519a05/workspace/VAE_ADHD/jubin_finetuning/psm_matched.csv')

        return psm.df_matched.copy()
    
    label_name = (args.cat_target+args.num_target)[0]
    n_outerCV = int(1/args.test_size)
    n_innerCV = 4

    # run propensity score matching & select one group which have matched_ID
    labels_fulldata = labels_total[labels_total.Mental_quotient != 0]
    df_matched = run_psm(labels=labels_fulldata, treatment='Autism', args=args)
    df_matched = df_matched[[args.subjectkey,'matched_ID']].dropna()
    label_kfold = labels_total[labels_total.subjectkey.isin(df_matched.subjectkey)]

    # do Iterative stratification
    folds = iter_strat(label_kfold, n_outerCV, args)

    # after getting k-folds, separate it into test split & train/valid split
    matched_dfs=[] # extract indices for each fold
    tv_idx = []
    for t_idx, v_idx in folds:
        psm_val_df = imageFiles_labels[imageFiles_labels.subjectkey.isin(label_kfold.iloc[v_idx].subjectkey)]# treatment group
        matched_val_df = imageFiles_labels[imageFiles_labels.subjectkey.isin(df_matched.iloc[v_idx].matched_ID)] # matched group
        matched_dfs.append(pd.concat([psm_val_df, matched_val_df]))

    # make test split
    for i in range(n_outerCV):
        print('-------------------------------------')
        print("Effect size for test fold",i)
        testlabel = labels_total[labels_total.subjectkey.isin(matched_dfs[i].subjectkey)]
        for var in ['sex', 'age(days)', 'Mental_quotient', 'Motor_quotient']:
            print(var,':',round(cohenD(testlabel, 'GDD', var),4))
    
    print(f"Use fold {args.test_fold} as test")
    df_test = matched_dfs.pop(args.test_fold)

    # make train/valid split
    args.num_folds = n_innerCV
    train_cv_dfs, val_cv_dfs = [None]*n_innerCV, [None]*n_innerCV
    if args.balanced_split == 'psm_testbalan_iter_strat':
        label_rest = labels_total[~labels_total.subjectkey.isin(df_test.subjectkey)]
        tv_idx = iter_strat(label_rest, n_innerCV, args)
        for i, (t_idx, v_idx) in enumerate(tv_idx):
            df_train = imageFiles_labels[imageFiles_labels.subjectkey.isin(label_rest.iloc[t_idx].subjectkey)]
            df_val = imageFiles_labels[imageFiles_labels.subjectkey.isin(label_rest.iloc[v_idx].subjectkey)]
            train_cv_dfs[i], val_cv_dfs[i] = df_train, df_val     
            print('-------------------------------------')
            print("Effect size for train/val fold",i)
            trainlabel = labels_total[labels_total.subjectkey.isin(df_train.subjectkey)]
            vallabel = labels_total[labels_total.subjectkey.isin(df_val.subjectkey)]
            for var in ['sex', 'age(days)', 'Mental_quotient', 'Motor_quotient']:
                print(var,':',round(cohenD(trainlabel, 'GDD', var),4))
                print(var,':',round(cohenD(vallabel, 'GDD', var),4))
    else:
        tv_idx = []
        for i in range(len(matched_dfs)):
            df_all = matched_dfs[:]
            df_val = df_all.pop(i)
            df_train = pd.concat(df_all)
            train_cv_dfs[i], val_cv_dfs[i] = df_train, df_val
            tv_idx.append((df_train.index, df_val.index))
            

    # check train/valid folds
    cols_to_see = []
    for col_list in [i for i in Config.iter_strat_label.values()]:
        cols_to_see = cols_to_see+col_list
    cols_to_see.append(label_name)
    cols_to_see = list(set(cols_to_see)) #remove redundancy
    get_info_fold(tv_idx, labels_total, cols_to_see)

    return train_cv_dfs, val_cv_dfs, df_test

    
def iterative_stratification(imageFiles_labels, labels_total, num_total, args):
    if args.val_size % args.test_size != 0:
        print("Validation size & Test size aren't matched to make folds. change val size to test size")
        args.val_size = args.test_size

    folds = iter_strat(labels_total, int(1/args.test_size), args)
    indices = [ x[1] for x in folds ] 

    # split dataset
    fold_test, fold_val = int(args.num_folds*args.test_size), int(args.num_folds*args.val_size)
    split_indices = indices[:fold_test], indices[fold_test:fold_test+fold_val], indices[fold_test+fold_val:]
    test_idx, val_idx, train_idx = [ np.concatenate(idx) for idx in split_indices ]
    get_info_fold(labels_total, train_idx, val_idx, test_idx, binary_target+continuous_target)

    num_train, num_val =len(train_idx), len(val_idx)
    new_idx = np.concatenate([train_idx, val_idx, test_idx])
    imageFiles_labels = imageFiles_labels.iloc[new_idx]
        
    return num_train, num_val, imageFiles_labels


def slice_index(array, n_chunks ):
    partitioned_list = np.array_split(np.sort(array), n_chunks)
    return [i[-1] for i in partitioned_list]
    

def multilabel_matrix_maker(df, binary_cols=None, multiclass_cols=None, continuous_cols=None, n_chunks=3) :

    """
    returns matrix that will be used for multilabel, taking into account columns that are either multiclass or continuous
    * df : the dataframe to be split
    * binary_cols : LIST of cols (str)just cols that will be used (binarized)
    * multiclass_cols : LIST of the cols (str) that are multi class
    * continuous_cols : LIST of the cols (str) that will be split (continouous)
    * n_chunks : if using continouous cols are used, how many split?
    
    outputs matrix that has binarized binarized for all columns (only needs to be used during iskf to get the indices)
    """
    df = df.copy() #copy becasue we don't want to modify the original df
    if binary_cols == multiclass_cols == continuous_cols == None : #i.e. if all are None
        raise ValueError("at least one of the cols have to be put.. currently all cols are selected as None")
    if type(binary_cols)!= list or type(multiclass_cols)!= list or type(continuous_cols)!= list: 
        raise ValueError("the cols have to be lists!")
    #checking if NaN exist => raise error (sanity check)\
    if df[binary_cols+multiclass_cols+continuous_cols].isnull().values.any():
        raise ValueError("Your provided df had some NaN in columns that you are wanting to do iskf on")    
 
    #now adding binarized columns for each column types and aggregating them into total_cols
    total_cols = []
    if binary_cols : 
        for col in binary_cols :
            df[col] = pd.factorize(df[col], sort = True)[0] 
            total_cols.append(df[col].values) #or single []?  ([[]] : df 로 만드는 것, [] : series로 만듬) 
            
    if multiclass_cols :
        for col in multiclass_cols : 
            df_col = df[[col]] #[[]] not [] because of dims 
            ohe = OneHotEncoder()
            ohe.fit(df_col)
            binarized_col = ohe.transform(df_col).todense() 
            total_cols.append(binarized_col)
            
    if continuous_cols: 
        for col in continuous_cols:
            df[col] = df[col].astype('float') #change to float when doing 
            array = df[col].values
            boundaries = slice_index(array, n_chunks)  
            i_below = -np.infty
            for i in boundaries:
                extracted_df = (df[col]>i_below) & (df[col]<=i) 
                i_below = i #update i_below
                total_cols.append(extracted_df.values.astype(float))     
    
    #adding all together,
    final_arr = np.column_stack(total_cols)
    
    return final_arr


def get_info_fold(kf_split, df, target_col): #get info from fold
    """
    * kf_split : the `kf.split(XX)`된것
    * df : the dataframe with the metadata I will use
    * target_col : the columns in the df that I'll take statistics of
    """
    train_dict = defaultdict(list)
    valid_dict = defaultdict(list)

    for FOLD, (train_idx, valid_idx) in enumerate(kf_split):
        label_train = df.iloc[train_idx]
        label_valid = df.iloc[valid_idx]
        for col in target_col:
            if df[col].nunique()<=10: # case: categorical variable
                keys=list(map(lambda x: f'{col}[{x}]', df[col].value_counts().index))
                train_counts=label_train[col].value_counts()
                valid_counts=label_valid[col].value_counts()
                for i, key in enumerate(keys):
                    try:
                        train_dict[key].append(train_counts.iloc[i])
                    except:
                        train_dict[key].append(0)
                    try:
                        valid_dict[key].append(valid_counts.iloc[i])
                    except:
                        valid_dict[key].append(0)
            else: # case: continuous variable
                train_dict[f'{col}-mean/std'].append(f'{label_train[col].mean():.2f} / {label_train[col].std():.2f}')
                valid_dict[f'{col}-mean/std'].append(f'{label_valid[col].mean():.2f} / {label_valid[col].std():.2f}')

    print("=== Fold-wise categorical values information of training set ===")
    print(pd.DataFrame(train_dict))
    print("=== Fold-wise categorical values information of validation set ===")
    print(pd.DataFrame(valid_dict))
    
    
def case_control_count(labels, dataset_type, args):
    if args.cat_target:
        df_labels = pd.DataFrame.from_records(labels)
        for cat_target in args.cat_target:
            curr_cnt = df_labels[cat_target].value_counts()
            print(f'In {dataset_type},\t"{cat_target}" contains {curr_cnt[1]} CASE and {curr_cnt[0]} CONTROL')
