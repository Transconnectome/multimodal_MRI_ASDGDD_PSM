## ======= load module ======= ##
import os
import glob
import time
import json
import random
import datetime
import hashlib
import argparse
from copy import deepcopy
from collections import defaultdict

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from config import Config
from utils.utils import seed_all, select_model, save_exp_result, checkpoint_load
from dataloaders.dataloaders import make_dataset, make_dataloaders
from envs.experiments import test
from envs.loss_functions import (calculating_loss_acc, calc_acc_auroc_ap, calc_MAE_MSE_R2)
import warnings
warnings.filterwarnings("ignore")

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system') # to prevent "Too many open files" error.

try:
    import torch._dynamo
    torch._dynamo.config.verbose=True
    torch._dynamo.config.suppress_errors = True
except:
    print("torch.dynamo import failed")
    
## ========= Helper Functions =============== ##    
class NewArgs(object): 
      
    def __init__(self, my_dict, args): 
        arg_dict = args.__dict__
        for key, value in my_dict.items(): 
            setattr(self, key, value) 
        
        for key, value in arg_dict.items():
            setattr(self, key, value) 
            
            
def test(net, dataset, args):
    def set_result(args):
        keys = ['loss','label','pred']
        if args.cat_target:
            keys.extend(['acc','auroc','ap'])
        if args.num_target:
            keys.extend(['abs_loss','mse_loss','r_square'])
        result = defaultdict(defaultdict, { k:defaultdict(None) for k in keys})
    
        return result
    
    def calc_confusion_matrix(confusion_matrices, curr_target, output, y_true):
        _, predicted = torch.max(output.data,1)
        tn, fp, fn, tp = confusion_matrix(y_true.numpy(), predicted.numpy()).ravel()
        confusion_matrices[curr_target]['True Positive'] = int(tp)
        confusion_matrices[curr_target]['True Negative'] = int(tn)
        confusion_matrices[curr_target]['False Positive'] = int(fp)
        confusion_matrices[curr_target]['False Negative'] = int(fn) 
    
    def seed_worker(worker_id):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             persistent_workers=args.persistent_workers,
                                             pin_memory=args.pin_memory,
                                             worker_init_fn=seed_worker,
                                             generator=g)
    net.eval()
    if hasattr(net, 'module'):
        device = net.device_ids[0]
    else: 
        device = 'cuda:0' if args.sbatch =='True' else f'cuda:{args.gpus[0]}'
    
    outputs = defaultdict(list)
    y_true = defaultdict(list)
    test_result = defaultdict(defaultdict)
    confusion_matrices = defaultdict(defaultdict)

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            image, targets = data if len(data) == 2 else (data, []) #230313change
            image = image.to('cuda')
            output = net(image)
            if 'MM' not in args.model or args.mode != 'pretraining':
                for curr_target in output:
                    if curr_target == 'embeddings':
                        continue
                    outputs[curr_target].append(output[curr_target].cpu())
                    y_true[curr_target].append(targets[curr_target].cpu())
            else:
                loss = calculating_loss_acc(targets, outputs, test_result, net, args)
    
    # caculating ACC and R2 at once  
    if 'MM' in args.model and args.mode == 'pretraining':
        test_acc = np.mean(test_result['loss'][args.metric])
        return test_acc, None
    # else:
    #     for key in ['loss','label','pred']:
    #         test_result.pop(key) 

    for curr_target in outputs:
        if curr_target == 'embeddings':
            continue
            
        outputs[curr_target] = torch.cat(outputs[curr_target])
        y_true[curr_target] = torch.cat(y_true[curr_target])
            
        acc_func = calc_acc_auroc_ap if curr_target in args.cat_target else calc_MAE_MSE_R2
        curr_acc = acc_func(outputs[curr_target], y_true[curr_target], args, None)
        for k in curr_acc:
            test_result[k][curr_target] = curr_acc[k]
        
        if curr_target in args.confusion_matrix:
            calc_confusion_matrix(confusion_matrices, curr_target,
                                  outputs[curr_target], y_true[curr_target])

    return test_result, confusion_matrices, outputs['Autism'], y_true['Autism']
    
    
## ========= Experiment =============== ##
def experiment(partition, subject_data, args):
    def run_report(net, dataset, args, result, split='test'):
        test_acc, confusion_matrices, outputs, y_true = test(net, dataset, args)
        subjects = list(dataset.image_files[args.data_type[0]].map(lambda x: x[x.index('sub-'):x.index('.')]))

        print("Result for each subject")
        df = pd.read_csv(Config.phenotype_dict[args.dataset][args.phenotype])
        scores = pd.DataFrame(columns=['subjectkey','probability','Autism'])
        for i in range(len(subjects)):
            scores.loc[i] = (subjects[i], torch.softmax(outputs[i],0)[1].item(), y_true[i].item())
        scores = pd.merge(scores, df[['subjectkey','sex','age(months)']], on='subjectkey', how='left')
        print(scores)
        csv_name = args.save_dir + '/scores/' + f"{args.exp_name}_fold{str(args.fold)}"
        csv_name = csv_name+'_'+split if split != 'test' else csv_name
        scores.to_csv(f'{csv_name}.csv')
        print('='*80)

        result[f'{split}_acc'] = test_acc
        print(f"===== {split} result for {args.exp_name} =====") 
        print(test_acc)

        if confusion_matrices != None:
            print("===== Confusion Matrices =====")
            print(confusion_matrices,'\n')
            result['confusion_matrices_'+split] = confusion_matrices
    
    # selecting a model
    device_ids = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
    net = select_model(subject_data, args)
    
    # loading pretrained model if transfer option is given
    print("*** Model setting for transfer learning & fine tuning *** \n")
    model_dir = glob.glob(f'result/model/*{args.load}*')[0]
    print(f"Loaded {model_dir[:-4]}")

    # setting for results' DataFrame
    result = defaultdict(list)
    
    print("\n*** Start testing a model *** \n")
    net = checkpoint_load(net, model_dir, args, test=True) #230313change
    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids = device_ids)        
    net.to(f'cuda:{device_ids[0]}')
    
    run_report(net, partition['test'], args, result)
    if args.save_train_val is not None:
        run_report(net, partition['train'], args, result, 'train')
        run_report(net, partition['val'], args, result, 'val')
        
    return vars(args), result
## ==================================== ##


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str)   
    parser.add_argument("--test_batch_size", type=int, default=4) 
    parser.add_argument("--save_train_val", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int) 
    args = parser.parse_args()
    
    ## ========= Setting ========= ##
    json_dir = glob.glob(f'result/*{args.load.split(".")[0]}*.json')[0]
    with open(json_dir, 'r') as f:
        setting = json.load(f)
    args = NewArgs(setting, args)
    args.save_dir = os.getcwd() + '/result'
    
    # Seed number
    seed_all(args.seed)
    
    ## ========= Run Experiment and saving result ========= ##    
    print(f"*** Experiment {args.exp_name} Start ***")
    partitions, subject_data = make_dataset(args)
    test_results = defaultdict(list)
    
    partition = partitions[args.fold]
    setting, result = experiment(partition, subject_data, deepcopy(args))
    for acc_type, acc_targets_dict in result['test_acc'].items():
        if acc_targets_dict.values():
            test_results[acc_type].extend(list(acc_targets_dict.values()))
            
    print(test_results)
                
    print("===== Experiment Setting Report =====")
    print(args)
    
    # Save result
    print("*** Experiment Done ***\n")
