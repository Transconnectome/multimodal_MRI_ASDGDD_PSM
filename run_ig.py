### load library
import os
import glob
import argparse
import json

import numpy as np
from tqdm.auto import tqdm # process bar

import torch
import torch.nn as nn

from dataloaders.dataloaders import make_dataset
from utils.utils import seed_all, checkpoint_load
from XAI.custom_attribution import NoiseTunnel, CustomIG
import XAI.models_wrapper as densenet3d 

## arguments
def argument_setting():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", type=str, default=None,required=True)
    parser.add_argument('--get_predicted_score', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(get_predicted_score=False)

    # Options for model setting
    parser.add_argument("--model", type=str, required=True, help='Select model. e.g. densenet3D121.',
                        choices=['densenet3D121', 'densenet3D169', 'densenet3D201',
                                 'densenet3D121MM'])
    parser.add_argument("--in_channels", default=1, type=int, help='')
    
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
                        help="option for additional transform - [crop, no_resize] are available")
    parser.add_argument("--augmentation", nargs="*", default=[], type=str, choices=['affine','flip'],
                        help="Data augmentation - [affine, flip] are available")
    parser.add_argument("--augm_prob", default=0.2, type=float, help='')

    # Hyperparameters for model training
    parser.add_argument("--lr", default=0.01, type=float, help='')
    parser.add_argument("--lr_adjust", default=0.01, type=float, help='')
    parser.add_argument("--epoch", type=int, help='')
    parser.add_argument("--epoch_FC", type=int, default=0, help='Option for training only FC layer')
    parser.add_argument("--optim", default='Adam', type=str, choices=['Adam','SGD','RAdam','AdamW'], help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='')
    parser.add_argument("--scheduler", default='', type=str, help='') 
    parser.add_argument("--early_stopping", default=None, type=int, help='')
    parser.add_argument("--early_stop_metric", default='auroc', type=str, help='')
    parser.add_argument("--num_workers", default=3, type=int, help='')
    parser.add_argument("--pin_memory", default=False, type=int, help='')
    parser.add_argument("--persistent_workers", default=True, type=int, help='')
    
    parser.add_argument('--accumulation_steps', default=None, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int, help='')
    parser.add_argument("--val_batch_size", default=16, type=int, help='')
    parser.add_argument("--test_batch_size", default=1, type=int, help='')
    parser.add_argument("--use_weighted_loss", type=str, nargs="*",
                        help='option that uses weighted loss when there is class imbalance')

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
    
    # Options for XAI setting
    parser.add_argument("--nt_batch_size", default=1, type=int, help='')
    parser.add_argument("--ig_only", default=None, type=str, help='')
    
    # Options for environment setting
    parser.add_argument("--exp_name", type=str, required=True, help='')
    parser.add_argument("--save_dir", type=str, help='')
    parser.add_argument("--split_seed", type=int, default=0, help='')
    parser.add_argument("--seed", type=int, default=1234, help='') 
    parser.add_argument("--gpus", nargs='+', type=int, help='')
    parser.add_argument("--sbatch", type=str, choices=['True', 'False'])
    parser.add_argument("--debug", default='', type=str, help='')

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args


def get_config(config_dir):
    with open(config_dir, 'r') as file:
        config = json.load(file)
    print("Configuration of IntegratedGrad is as follow.{}".format(config))
    return config


def get_subject_list(dataset_subjid):
    subject_list = []
    for i, subj in enumerate(dataset_subjid):
        _, subj = os.path.split(subj)
        if subj.find('.nii.gz') != -1: 
            subj = subj.replace('.nii.gz','')
        elif subj.find('.npy') != -1: 
            subj = subj.replace('.npy','')
        subject_list.append(subj) 
    return subject_list


def save_attribute(attr_outputs: np.ndarray, subject_list: list, target_name, save_dir, args): 
    attr_save_dir = save_dir
    os.makedirs(attr_save_dir, exist_ok=True)
    print(len(attr_outputs), len(subject_list))
    assert len(attr_outputs) == len(subject_list)
    for i, subject_id in enumerate(subject_list):
        file_name = os.path.join(attr_save_dir, subject_id + '.npy')
        np.save(file_name, attr_outputs[i])
    

def XAI_engine(interpreter, dataset, ig_config, args):
    testloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.test_batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=args.num_workers)    

    attr_outputs = {}
    #ig_config['nt_samples']=2 # change here
    for target in args.cat_target+args.num_target:
        attr_outputs[target] = torch.tensor([])
        
    for idx, data in enumerate(tqdm(testloader,0)):
        images, targets = data if len(data) == 2 else (data, []) 
        # images = tuple(map(lambda x: torch.Tensor(x).cuda(rank), images))# [0] # added
        images = images.cuda()
        
        for target in args.cat_target+args.num_target: 
            ig_targets = targets[target].cuda() if target in args.cat_target else None
            if args.ig_only is not None:
                 attr = interpreter.attribute(inputs=images,
                                             target=ig_targets,
                                             internal_batch_size=args.test_batch_size)
            else:
                attr = interpreter.attribute(inputs=images,
                                             target=ig_targets,
                                             internal_batch_size=args.test_batch_size,
                                             nt_samples=ig_config['nt_samples'],
                                             nt_samples_batch_size=args.nt_batch_size,
                                             stdevs=ig_config['stdevs'],
                                             nt_type=ig_config['nt_type'])
            attr_outputs[target] = torch.cat([attr_outputs[target], attr.detach().cpu()])

    for target in attr_outputs:
        attr_outputs[target] = attr_outputs[target].squeeze().numpy()

    return attr_outputs


def XAI_experiments(partition, subject_data, save_dir, config_dir, args):
    targets = args.cat_target + args.num_target
    args.study_sample = args.dataset

    # DenseNet
    if args.model == 'densenet3D121':
        net = densenet3d.densenet121(subject_data=subject_data, args=args, mode='classifier', drop_rate=0.0)
    elif args.model == 'densenet3D161':
        net = densenet3d.densenet3D161(subject_data, args) 
    elif args.model == 'densenet3D169':
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        net = densenet3d.densenet3D201(subject_data, args)

    # load checkpoint and attach module to GPU 

    assert args.checkpoint_dir
    model_root_dir = f'{current_dir}/result/model'
    model_dir = glob.glob(f'{model_root_dir}/*{args.checkpoint_dir}*')[0]
            
    try:
        net = checkpoint_load(net, model_dir, args)
        if torch.cuda.device_count() > 0:
            net = nn.DataParallel(net, device_ids = list(range(torch.cuda.device_count()))) 
    except:
        if torch.cuda.device_count() > 0:
            net = nn.DataParallel(net, device_ids = list(range(torch.cuda.device_count())))
        net = checkpoint_load(net, model_dir, args)
        
    # attach network module to gpu
    net.cuda()
    net.eval()

    ig_config = get_config(config_dir=config_dir)
    
    # setting for Integrated Grad
    interpreter = CustomIG(net)
    interpreter = NoiseTunnel(interpreter) # if you don't want to use noise tunneling, plz deactivate this line

    # getting feature map 
    dataset = partition['test']
    subject_list = get_subject_list(dataset.image_files[args.data_type[0]])
    attr_outputs = XAI_engine(interpreter=interpreter, dataset=dataset, ig_config=ig_config, args=args)
    
    for target in targets:
        save_attribute(attr_outputs=attr_outputs[target], subject_list=subject_list, target_name=target, save_dir=save_dir, args=args)
    
      
        
if __name__ == "__main__":
    ## ========= Setting ========= ##
    args = argument_setting()
    attribute_dir = f'/result/attribute/{args.exp_name}'
    current_dir = os.getcwd()
    save_dir = current_dir+attribute_dir if not args.save_dir else args.save_dir+attribute_dir
    config_dir = current_dir + '/XAI/config.json'
    
    # Seed number
    args.seed = 1234
    seed_all(args.seed)

    ## ========= Run Experiment and saving result ========= ##    
    print(f"*** Experiment {args.exp_name} Start ***")
    partitions, subject_data = make_dataset(args)
    XAI_experiments(partitions[args.test_fold], subject_data, save_dir, config_dir, args)