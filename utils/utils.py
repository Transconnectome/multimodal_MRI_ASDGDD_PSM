import pdb
import os
import random
import json
import argparse 
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch

import models.densenet3d as densenet3d
from models.densenetYA import densenet121

def argument_setting():
    parser = argparse.ArgumentParser()

    # Options for model setting
    parser.add_argument("--model", type=str, required=True, help='Select model. e.g. densenet3D121.')
    parser.add_argument("--in_channels", default=1, type=int, help='')
    parser.add_argument("--dropout", default=0, type=float, help='')
    
    # Options for dataset and data type, split ratio, CV, resize, augmentation
    parser.add_argument("--dataset", type=str, choices=['CHA'], required=True, help='Select dataset') 
    parser.add_argument("--data_type", nargs='+', type=str, help='Select data type(sMRI, dMRI)')
    parser.add_argument("--multimodal", type=str) 
    parser.add_argument("--phenotype", default='total', type=str, help='')
    parser.add_argument("--balanced_split", default='', type=str, help='')
    parser.add_argument("--N", default=None, type=int, help='')
    parser.add_argument("--tissue", default=None, type=str, help='Select tissue mask(Cortical grey matter, \
                        Sub-cortical grey matter, White matter, CSF, Pathological tissue)',
                        choices=['cgm', 'scgm', 'wm', 'csf', 'pt'])
    parser.add_argument("--metric", default='', type=str, help='')
    parser.add_argument("--val_size", default=0.1, type=float, help='')
    parser.add_argument("--test_size", default=0.1, type=float, help='')
    parser.add_argument("--cv", default=None, type=str, help="option for splitting K-fold CV manually.")
    parser.add_argument("--nested_cv", default=False, type=bool, help="if True, then run through inner loop")
    parser.add_argument("--test_fold", default=0, type=int, help='which fold will be used for test set used in nested_CV')
    parser.add_argument("--resize", nargs="*", default=(96, 96, 96), type=int, help='')
    parser.add_argument("--transform", nargs="*", default=[], type=str, 
                        help="option for additional transform - [crop, no_resize, pad, 2mm] are available")
    parser.add_argument("--augmentation", nargs="*", default=[], type=str, choices=['affine','flip'],
                        help="Data augmentation - [affine, flip] are available")
    parser.add_argument("--augm_prob", default=0.2, type=float, help='')
    parser.add_argument("--num_workers", default=3, type=int, help='')
    parser.add_argument("--pin_memory", default=False, type=int, help='')
    parser.add_argument("--persistent_workers", default=True, type=int, help='')
    

    # Hyperparameters for model training
    parser.add_argument("--lr", default=0.01, type=float, help='')
    parser.add_argument("--lr_adjust", default=0.01, type=float, help='')
    parser.add_argument("--epoch", type=int, required=True, help='')
    parser.add_argument("--epoch_FC", type=int, default=0, help='Option for training only FC layer')
    parser.add_argument("--optim", default='Adam', type=str, choices=['Adam','SGD','RAdam','AdamW'], help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='')
    parser.add_argument("--scheduler", default='', type=str, help='') 
    parser.add_argument("--early_stopping", default=None, type=int, help='')
    parser.add_argument("--early_stop_metric", default='auroc', type=str, help='')
    parser.add_argument('--accumulation_steps', default=None, type=int, required=False)
    parser.add_argument("--train_batch_size", default=16, type=int, help='')
    parser.add_argument("--val_batch_size", default=16, type=int, help='')
    parser.add_argument("--test_batch_size", default=1, type=int, help='')
    parser.add_argument("--use_weighted_loss", type=str, nargs="*",
                        help='option that uses weighted loss when there is class imbalance')
    parser.add_argument("--use_label_smoothing", type=float, default=0)
    parser.add_argument("--use_negative_loss", nargs='+', default=[], type=str, help='')   
    parser.add_argument("--adjust_threshold", default=None, type=str, help='')   
    

    # Options for experiment setting
    parser.add_argument("--cat_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_target", nargs='+', default=[], type=str, help='')
    parser.add_argument("--num_normalize", type=str, default=True, help='')
    parser.add_argument("--confusion_matrix",  nargs='*', default=[], type=str, help='')
    parser.add_argument("--filter", nargs="*", default=[], type=str,
                        help='options for filter data by phenotype. usage: --filter sex:1')
    parser.add_argument("--mode", default='pretraining', type=str,  choices=['pretraining','finetuning','transfer'],
                        help='Option for learning from scratch')
    parser.add_argument("--load", default='', type=str, help='Load model weight that mathces {your_exp_dir}/result/*{load}*')
    parser.add_argument("--unfrozen_layers", default='all', type=str, help='Select the number of layers that would be unfrozen')
    parser.add_argument("--init_unfrozen", default='', type=str, help='Initializes unfrozen layers')
    parser.add_argument("--confounder", default='age(days)', type=str, help='')
    
    # Options for environment setting
    parser.add_argument("--exp_name", type=str, required=True, help='')
    parser.add_argument("--save_dir", type=str, help='')
    parser.add_argument("--seed", type=int, default=1234, help='')
    parser.add_argument("--split_seed", type=int, default=0, help='')
    parser.add_argument("--wandb", type=str, default=None, help='')    
    parser.add_argument("--gpus", type=int, nargs="*", default=[], help='')
    parser.add_argument("--sbatch", type=str, choices=['True', 'False'])
    parser.add_argument("--debug", default='', type=str, help='')
        
    args = parser.parse_args()
    if ((args.cat_target + args.num_target) == []) and 'MM' not in args.model:
        raise ValueError('--num-target or --cat-target should be specified')
        
    print(f"*** Categorical target labels are {args.cat_target} and Numerical target labels are {args.num_target} *** \n")

    return args


def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    
# Return a specific model based on argument setting
def select_model(subject_data, cf_kernel, dataset_size, args):
    # DenseNet-121 (primary architecture)
    if 'densenet3D121' in args.model:
        net = densenet121(subject_data=subject_data, args=args, mode='classifier', drop_rate=0.0)
    elif args.model == 'densenet3D169':
        net = densenet3d.densenet3D169(subject_data, args)
    elif args.model == 'densenet3D201':
        net = densenet3d.densenet3D201(subject_data, args)
    else:
        raise ValueError(f"Unsupported model: {args.model}. Use densenet3D121/169/201.")

    return net

# Experiment 
def CLIreporter(train_loss, train_acc, val_loss, val_acc):
    '''command line interface reporter per every epoch during experiments'''
    print("="*80)
    visual_report = defaultdict(list)
    for label_name in train_loss:
        loss_value = f'{train_loss[label_name]:2.4f} / {val_loss[label_name]:2.4f}'
        if 'contrastive_loss' not in label_name:
            acc_value = f'{train_acc[label_name]:2.4f} / {val_acc[label_name]:2.4f}' 
        else:
            acc_value = None
        visual_report['Loss (train/val)'].append(loss_value)
        visual_report['R2 or ACC (train/val)'].append(acc_value)
    print(pd.DataFrame(visual_report, index=train_loss.keys()))
    
    return None


# define checkpoint-saving function
def checkpoint_save(net, epoch, args):
    """checkpoint is saved only if validation performance for all target tasks are improved """
    if args.debug != '':
        return None
    if os.path.isdir(os.path.join(args.save_dir, 'model')) == False:
        makedir(os.path.join(args.save_dir, 'model'))
    
    checkpoint_dir = os.path.join(args.save_dir, f'model/{args.model}_{args.exp_name}_fold{args.fold}.pth')
    if epoch == args.epoch:
        checkpoint_dir = checkpoint_dir[:-4]+"_last.pth"
    torch.save(net.module.state_dict(), checkpoint_dir)

    return checkpoint_dir


def checkpoint_load(net, checkpoint_dir, args, test=False): #230313change
    if hasattr(net, 'module'):
        net = net.module

    model_state = torch.load(checkpoint_dir, map_location = 'cpu')
    if 'MM' in args.model and args.mode != 'pretraining' and test == False:
        try:
            net.load_state_dict(model_state, strict=True)
        except:
            extractor_state = OrderedDict()
            for k, v in model_state.items():
                if 'feature_extractors' not in k:
                    break
                new_k = '.'.join(k.split('.')[1:])
                extractor_state[new_k] = model_state[k]
            net.feature_extractors.load_state_dict(extractor_state)
    else:
        net.load_state_dict(model_state, strict=True)
    
    print('The best checkpoint is loaded')

    return net
    
    
# define result-saving function
def save_exp_result(setting, result):
    if setting['debug']=='1':
        return None
    makedir(setting['save_dir'])
    exp_name = setting['exp_name']
    del setting['epoch']
    del setting['test_batch_size']

    filename = setting['save_dir'] + f'/{exp_name}_fold{setting["fold"]}.json'
    result.update(setting)

    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)

def print_test_results(num_folds, result, args):
    text = "Metric"+''.join([*map(lambda x: f"\tfold{x}", range(num_folds))])
    for acc_type in result:
        text += '\n' + acc_type + ''.join([ f'\t{x:.4f}' for x in result[acc_type] ])
    print(text)    
    return text
        
        
def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
