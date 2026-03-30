## ======= load module ======= ##
import os
import pdb
import glob
import time
import datetime
import hashlib
from copy import deepcopy
from collections import defaultdict

import wandb
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.optimizer import CosineAnnealingWarmupRestarts
from utils.utils import (argument_setting, seed_all, select_model,
                         CLIreporter, save_exp_result, print_test_results,
                         checkpoint_save, checkpoint_load)
from dataloaders.dataloaders import make_dataset, make_dataloaders
from envs.experiments import train, validate, test

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

# torch.autograd.set_detect_anomaly(True)

## ========= Helper Functions =============== ##
def setup_results(args):
    result = defaultdict(list)
    result['train_losses'] = defaultdict(list)
    result['train_accs'] = defaultdict(list)
    result['val_losses'] = defaultdict(list)
    result['val_accs'] = defaultdict(list)
    
    return result

    
def set_optimizer(args, net):
    if args.scheduler.lower() == 'cos':
        init_lr = 1e-9
    else:
        init_lr = args.lr
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(params = filter(lambda p: p.requires_grad, net.parameters()),
                              lr=init_lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, net.parameters()),
                               lr=init_lr, weight_decay=args.weight_decay)
    elif args.optim =='RAdam':
        optimizer = optim.RAdam(params = filter(lambda p: p.requires_grad, net.parameters()),
                                lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(params = filter(lambda p: p.requires_grad, net.parameters()),
                                lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer choice')
        
    return optimizer
    
    
def set_lr_scheduler(args, optimizer, len_dataloader):
    if args.scheduler == '':
        scheduler = None
    elif args.scheduler == 'on':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=10, factor=0.2, min_lr=1e-7)
    elif args.scheduler == 'cos' or 'cawr' in args.scheduler:
        if args.scheduler.endswith('2'):
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, cycle_mult=2,
                                                  max_lr=args.lr, min_lr=1e-9, warmup_steps=5, gamma=1)
        else:
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=40, cycle_mult=1,
                                                  max_lr=args.lr, min_lr=1e-9, warmup_steps=10, gamma=0.75)
    elif 'anneal' in args.scheduler.lower():
        t_max = 50 if 'decay' not in args.scheduler.lower() else args.epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif 'step' in args.scheduler:
        step_size = 80 if len(args.scheduler.split('_')) != 2 else int(args.scheduler.split('_')[1])        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    elif args.scheduler.lower() == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epoch)
    else:
        raise Exception("Invalid scheduler option")
        
    return scheduler
    
    
def add_epoch_result(result, train_result, val_result, metric): #230313change
    loss_acc_sum = defaultdict(int)
    for target_name in train_result['loss']:
        result['train_losses'][target_name].append(train_result['loss'][target_name])
        result['val_losses'][target_name].append(val_result['loss'][target_name])
        loss_acc_sum['train_loss'] += train_result['loss'][target_name]
        loss_acc_sum['val_loss'] += val_result['loss'][target_name]

        for acc in train_result:
            if 'MM' in args.model and 'contrastive_loss' in target_name:
                loss_acc_sum[f'val_metric'] += -val_result[acc][target_name]
            if ('contrastive_loss' not in target_name) and acc != 'loss' and target_name in train_result[acc]:
                loss_acc_sum[f'train_{acc}'] += train_result[acc][target_name]
                loss_acc_sum[f'val_{acc}'] += val_result[acc][target_name]
                if acc in metric:
                    loss_acc_sum[f'val_metric'] += val_result[acc][target_name]
                    result[f'train_accs'][target_name].append(train_result[acc][target_name])
                    result[f'val_accs'][target_name].append(val_result[acc][target_name])
    
    return loss_acc_sum


def calculate_confounder_kernel(dataloader, args):
    N = len(dataloader.dataset)
    labels = dataloader.dataset.subject_metadata['Autism'].to_numpy()
    cf = dataloader.dataset.subject_metadata[args.confounder].to_numpy()
    
    X = np.zeros((N,3))
    X[:,0] = labels
    X[:,1] = cf
    X[:,2] = np.ones((N,))
    XTX = np.transpose(X).dot(X)
    kernel = np.linalg.inv(XTX)
    cf_kernel = nn.Parameter(torch.tensor(kernel).float().to('cuda:0'), requires_grad=False)
    
    return cf_kernel

    
## ========= Train & Validate =============== ##
def run_experiment(args, partition, result, mode):
    # Make dataloaders
    trainloader, valloader = make_dataloaders(partition, args)

    # calculate confounder kernel for MetaDataNormalization Layer
    if 'mdn' in args.model.lower():
        cf_kernel = calculate_confounder_kernel(trainloader, args)
    else:
        cf_kernel = None

    # selecting a model
    net = select_model(subject_data, cf_kernel, len(trainloader.dataset), args)

    # setting a DataParallel and model on GPU
    
    net = nn.DataParallel(net, device_ids = args.gpus)   
    net.to(f'cuda:{net.device_ids[0]}')
    if args.wandb:
        wandb.watch(net)
        
    # Setting for transfer learning or fine-tuning
    epoch_exp = args.epoch 

    # Set AMP, optimizer, scheduler
    scaler = torch.cuda.amp.GradScaler()
    optimizer = set_optimizer(args, net)
    scheduler = set_lr_scheduler(args, optimizer, len(partition['train']))

    # Setup experiment results variables
    best_loss_acc = {'train_loss': float('inf'), 'val_loss': float('inf'), 'val_metric': -float('inf')}
    checkpoint_dir = None
    patience = 0
    metric = args.early_stop_metric

    # Train & Validate model
    for epoch in tqdm(range(epoch_exp)):
        ts = time.time()
        curr_lr = optimizer.param_groups[0]['lr']
        net, train_result = train(net, trainloader, optimizer, scaler, args)
        val_result = validate(net, valloader, scheduler, args)

        ## sorting the results
        loss_acc_sum = add_epoch_result(result, train_result, val_result, metric)
        if args.wandb:
            wandb.log(data=dict(loss_acc_sum, **{'learning_rate':curr_lr}), step=epoch+1)

        ## Check if best epoch, save the checkpoint and results, visualize the result.
        is_best = (loss_acc_sum['val_loss'] < best_loss_acc['val_loss']) if args.early_stop_metric == 'loss' else (loss_acc_sum['val_metric'] > best_loss_acc['val_metric'])
        if is_best:
            result['best_epoch'] = epoch
            best_loss_acc.update(loss_acc_sum)
            checkpoint_dir = checkpoint_save(net, epoch, args)
            if args.wandb:
                wandb.summary.update({'best_'+k:v for k, v in best_loss_acc.items()})
                wandb.summary['best_epoch'] = epoch+1
        patience = (patience+1) * (not is_best)
        # save_exp_result(vars(args).copy(), result) 

        te = time.time()

        print(f"Epoch {epoch+1}. Loss: {loss_acc_sum['train_loss']:.4f} / {loss_acc_sum['val_loss']:.4f}.\
        Metric: {train_result['ap']['Autism']} / {val_result['ap']['Autism']}. \
        Lr: {curr_lr:.4e}. {te-ts:2.2f} sec. {is_best*'Best epoch'}")

        ## Early-Stopping
        if args.early_stopping != None and patience >= args.early_stopping and (result['best_epoch'] >= 20 or epoch < 100):
            print(f"*** Validation Loss patience reached {args.early_stopping} epochs. Early Stopping Experiment ***")
            break
                
    if args.debug:
        return result, None
    
    checkpoint_save(net, args.epoch, args)

    opt = '' if mode == 'ALL' else '_FC'
    result[f'best_val_loss{opt}'] = best_loss_acc['val_loss']
    result[f'best_train_loss{opt}'] = best_loss_acc['train_loss']

    return net, result, checkpoint_dir


## ========= Experiment =============== ##
def experiment(partition, subject_data, args):   
    # setting for results' DataFrame
    result = setup_results(args)
    
    # training a model
    print("*** Start training a model *** \n")
    net, result, checkpoint_dir = run_experiment(args, partition, result, 'ALL')
                    
    # testing a model
    if args.debug:
        return vars(args), result
    
    print("\n*** Start testing a model *** \n")
    net.to('cpu')
    torch.cuda.empty_cache()

    net = checkpoint_load(net, checkpoint_dir, args, test=True) #230313change
    if not isinstance(net, nn.DataParallel):
        net = nn.DataParallel(net, device_ids = args.gpus)   
    net.to(f'cuda:{args.gpus[0]}')
        
    test_acc, confusion_matrices = test(net, partition, args)
    result['test_acc'] = test_acc
    print(f"===== Test result for {args.exp_name} =====") 
    for k, v in test_acc.items():
        try:
            print(k, tuple(*v.items()))
        except:
            print(k, *tuple(v.items()))
        if args.wandb:
            try:
                wandb.summary[f'{k}'] = tuple(v.values())[-1] # tuple(*v.values())[0]
            except:
                wandb.summary[f'{k}'] = list(v.values())[-1]

    if confusion_matrices != None and args.cat_target:
        print("===== Confusion Matrices =====")
        print(confusion_matrices,'\n')
        result['confusion_matrices'] = confusion_matrices
        
    return vars(args), result
## ==================================== ##


if __name__ == "__main__":
    ## ========= Setting ========= ##
    args = argument_setting()
    args.save_dir = os.getcwd() + '/result' if not args.save_dir else args.save_dir
    
    # Seed number
    seed_all(args.seed)
    
    # Set exp_name
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'

    ## ========= Run Experiment and saving result ========= ##    
    print(f"*** Experiment {args.exp_name} Start ***")
    partitions, subject_data = make_dataset(args)
    test_results = defaultdict(list)
    for fold in range(len(partitions)):
        # Initialize wandb
        args.fold = fold
        if args.wandb:
            # Use environment variable for security: export WANDB_API_KEY=<your_key>
            wandb_key = os.environ.get('WANDB_API_KEY')
            if not wandb_key:
                raise ValueError("WANDB_API_KEY environment variable not set. Please set it with: export WANDB_API_KEY=<your_key>")
            wandb.login(key=wandb_key)
            project_name = f'{args.dataset}_Autism' if 'Autism' in args.cat_target else f'{args.dataset}_{"".join(args.cat_target+args.num_target)}'
            wandb.init(project=project_name,
                       group=f'{"&".join(args.data_type)}_testfold{args.test_fold}_testsize{args.test_size}_{args.multimodal}',
                       name=f'fold{fold}_{args.model}_{args.exp_name}', config=args)
    
        # Run Experiment
        partition = partitions[fold]
        setting, result = experiment(partition, subject_data, deepcopy(args))
        for acc_type, acc_targets_dict in result['test_acc'].items():
            if acc_targets_dict.values():
                test_results[acc_type].extend(list(acc_targets_dict.values()))
        if args.wandb:
            wandb.finish()
        if args.nested_cv == 1:
            break # change needed
        save_exp_result(setting, result)
        
    result['test_summary'] = print_test_results(num_folds=len(partitions), result=test_results, args=args)
    calc_best = min if args.early_stop_metric == 'loss' else max
    best_metric = calc_best(test_results['auroc'])
    result['best_fold'] = test_results['auroc'].index(best_metric)
    print("Best fold:", result['best_fold'])
    print("===== Experiment Setting Report =====")
    print(args)
    
    # Save result
    if args.debug:
        quit()
    
    save_exp_result(setting, result)
    print("*** Experiment Done ***\n")
