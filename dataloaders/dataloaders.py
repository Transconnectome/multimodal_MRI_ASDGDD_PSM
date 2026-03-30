import glob
import random
from collections import defaultdict
import pdb

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from monai.data import ImageDataset
from monai.transforms import (AddChannel, Compose, CenterSpatialCrop, Flip,
                              RandAffine, RandFlip, RandRotate90, Resize,
                              SpatialPad, ScaleIntensity, ToTensor)

from config import Config
from dataloaders.data_utils import psm_iterative_stratification, iterative_stratification, case_control_count
from dataloaders.custom_transform import MaskTissue
from dataloaders.custom_dataset import MultiModalImageDataset
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

def loading_images(image_dir, args):
    image_files = pd.DataFrame()
    data_types = args.data_type if (args.tissue == None) else args.data_type + ['5tt_warped_nii']
    for brain_modality in data_types:
        curr_dir = image_dir[brain_modality]
        curr_files = pd.DataFrame({brain_modality:glob.glob(curr_dir+'*[yz]')}) # to get .npy(sMRI) & .nii.gz(dMRI) files
        curr_files[subjectkey] = curr_files[brain_modality].map(lambda x: x.split("/")[-1].split('.')[0])
        curr_files.sort_values(by=subjectkey, inplace=True)
        
        if len(image_files) == 0:
            image_files = curr_files
        else:
            image_files = pd.merge(image_files, curr_files, how='inner', on=subjectkey)
            
    if args.debug:
        image_files = image_files[:160]
        
    return image_files


def loading_phenotype(phenotype_dir, target_list, args):
    col_list = target_list + [subjectkey]
    if 'multitarget' in args.balanced_split:
        col_list.append('split')

    ## get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    raw_csv = subject_data.copy()
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by=subjectkey)
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True)

    ### preprocessing categorical variables and numerical variables
    subject_data = preprocessing_cat(subject_data, args)
    if args.num_normalize == True:
        subject_data = preprocessing_num(subject_data, args)
    
    return subject_data, raw_csv 


def make_balanced_testset(il, raw_merged, num_total, args):
    if 'iter_strat' in args.balanced_split: # we need raw csv which have all phenotype data
        num_train, num_val, imageFiles_labels = iterative_stratification(il, raw_merged, num_total, args)
        
    elif 'multitarget' in args.balanced_split:
        num_train, num_val = il['split'].value_counts()[['train','val']]
        splits = [il[il['split']=='train'], il[il['split']=='val'], il[il['split']=='test']]
        imageFiles_labels = pd.concat(splits)
        
    return num_train, num_val, imageFiles_labels


# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels, raw_merged ,target_list, args, return_dfs=False):
    if args.N != None:
        imageFiles_labels = imageFiles_labels.sample(n=args.N).reset_index(drop=True)
        raw_merged = pd.merge(raw_merged, imageFiles_labels, on=subjectkey, how='right',suffixes=('','x'))
    ## Dataset split    
    num_total = len(imageFiles_labels) 
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size) 
    num_test = int(num_total*args.test_size)
     
    if 'psm' in args.balanced_split:
        train_cv_dfs, val_cv_dfs, il_test = psm_iterative_stratification(imageFiles_labels, raw_merged, args)
    
    ## Define transform function
    resize = tuple(args.resize)
    spatial_size = resize
    default_transforms = [ScaleIntensity(), AddChannel(), Resize(resize), ToTensor()] 
    
    if 'no_resize' in args.transform:
        default_transforms.pop(2)
        spatial_size = (256,256,256) if 'crop' not in args.data_type[0] else (206, 206, 206)
        if 'pad' in args.transform:
            default_transforms.insert(2, SpatialPad(spatial_size=(206, 206, 206)))
        if '2mm' in args.transform:
            spatial_size = (103,103,103)
            default_transforms.insert(3, Resize(spatial_size))
        if '1.5mm' in args.transform:
            spatial_size = (138,138,138)
            default_transforms.insert(3, Resize(spatial_size))
    if 'resize128' in args.data_type[0]:
        default_transforms.pop(2)
        spatial_size = (128,128,128)
    if 'crop' in args.transform:
        default_transforms.insert(2, CenterSpatialCrop(192))
    if args.tissue:
        dMRI_transform = [MaskTissue(imageFiles_labels['5tt_warped_nii'], args.tissue)]
        dMRI_transform += default_transforms
        
    aug_transforms = []
    if 'affine' in args.augmentation:
        aug_transforms.append(RandAffine(prob=args.augm_prob, padding_mode='zeros', # prob=0.2, padding_mode='zeros',
                                         translate_range=(np.array(spatial_size)*0.1), # translate_range=(np.array(spatial_size)*0.1),
                                         rotate_range=(np.pi/12,)*3, # rotate_range=(np.pi/36,)*3,
                                         spatial_size=spatial_size, cache_grid=True)) # spatial_size=spatial_size, cache_grid=True))
    elif 'flip' in args.augmentation:
        aug_transforms.append(RandFlip(prob=0.2, spatial_axis=0))
    
    train_transforms, val_transforms, test_transforms = [], [], []
    for brain_modality in args.data_type:
        curr_transform = dMRI_transform if args.tissue else default_transforms
        train_transforms.append(Compose(curr_transform + aug_transforms))
        val_transforms.append(Compose(curr_transform))
        test_transforms.append(Compose(curr_transform))
    
    ## make splitted dataset
    partitions, partition_dfs = [], []
    for il_train, il_valid in zip(train_cv_dfs, val_cv_dfs):
        train_set = MultiModalImageDataset(image_files=il_train[args.data_type],
                                           labels=il_train[target_list].to_dict('records'),
                                           subject_metadata=il_train[Config.confounders+target_list],
                                           confounder=il_train[args.confounder],
                                           transform=train_transforms)
        val_set = MultiModalImageDataset(image_files=il_valid[args.data_type],
                                         labels=il_valid[target_list].to_dict('records'),
                                         subject_metadata=il_valid[Config.confounders+target_list],
                                         confounder=il_valid[args.confounder],
                                         transform=val_transforms)
        test_set = MultiModalImageDataset(image_files=il_test[args.data_type],
                                          labels=il_test[target_list].to_dict('records'),
                                          subject_metadata=il_test[Config.confounders+target_list],
                                          confounder=il_test[args.confounder],
                                          transform=test_transforms)
        partitions.append({'train': train_set, 'val': val_set, 'test': test_set})
        if return_dfs:
            partition_dfs.append((il_train, il_valid, il_test))
        if args.use_weighted_loss:
            args.class_weights = dict()
            for target in args.use_weighted_loss:
                weights = [ 1 - x for x in (il_train[target].value_counts()[sorted(il_train[target].unique())]/len(il_train)) ]
                args.class_weights[target] = weights # shouldn't it be "args.class_weights[target][curr_fold] = weights"...??
                
    # case_control_count(labels_train, 'train', args)
    # case_control_count(labels_val, 'validation', args)
    # case_control_count(labels_test, 'test', args)

    return partitions, partition_dfs


def setting_dataset(args):
    subjectkey = 'subjectkey'
    image_dir = Config.data_dict[args.dataset]
    phenotype_dir = Config.phenotype_dict[args.dataset]
    phenotype_csv = phenotype_dir[args.phenotype]

    return subjectkey, image_dir, phenotype_csv


def make_dataset(args, return_dfs=False):
    global subjectkey
    subjectkey, image_dir, phenotype_dir = setting_dataset(args)
    args.subjectkey = subjectkey
    target_list = args.cat_target + args.num_target
    
    image_files = loading_images(image_dir, args)
    subject_data, raw_csv = loading_phenotype(phenotype_dir, target_list, args)

    # combining image files & labels
    raw_merged = pd.merge(raw_csv, image_files, how='inner', on=subjectkey)
    if 'multitarget' in args.balanced_split:
        imageFiles_labels = (pd.merge(subject_data, image_files, how='left', on=subjectkey)).dropna()
    else:
        imageFiles_labels = pd.merge(subject_data, image_files, how='inner', on=subjectkey)
    
    # merge confounder data to dataframe
    imageFiles_labels[Config.confounders] = raw_merged[Config.confounders]
    
    # partitioning dataset and preprocessing (change the range of categorical variables and standardize numerical variables)
    partition, dfs = partition_dataset(imageFiles_labels, raw_merged, target_list, args, return_dfs)
    print("*** Making a dataset is completed *** \n")
    
    if return_dfs:
        return partition, subject_data, dfs
    else:
        return partition, subject_data


def make_dataloaders(partition, args):
    def seed_worker(worker_id):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
        
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                              batch_size=args.train_batch_size,
                                              shuffle=True,
                                              persistent_workers=args.persistent_workers,
                                              num_workers=args.num_workers,
                                              pin_memory=args.pin_memory,
                                              worker_init_fn=seed_worker,
                                              generator=g)
    
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            persistent_workers=args.persistent_workers,
                                            num_workers=args.num_workers,
                                            pin_memory=args.pin_memory,
                                            worker_init_fn=seed_worker,
                                            generator=g)   
    
    return trainloader, valloader
