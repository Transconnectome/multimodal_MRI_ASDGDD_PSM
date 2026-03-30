import pdb
import random
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from envs.loss_functions import (calculating_loss_acc, calc_acc_auroc_ap,
                                 calc_MAE_MSE_R2, calc_dcor)

### ========= Helper functions ========= ###
def set_result(args):
    keys = ['loss','label','pred', 'embeddings', 'confounders']
    if args.cat_target:
        keys.extend(['acc','auroc','ap'])
    if args.num_target:
        keys.extend(['abs_loss','mse_loss','r_square'])
    result = defaultdict(defaultdict, { k:defaultdict(list) for k in keys})
    
    return result


def total_loss_acc(result, args, test=False, cfm=None):
    for target in result['loss']:
        result['loss'][target] = np.mean(result['loss'][target])
        if 'contrastive' in target:
            continue
        acc_func = calc_acc_auroc_ap if target in args.cat_target else calc_MAE_MSE_R2
        curr_label = torch.cat(result['label'][target]).cpu()
        curr_pred = torch.cat(result['pred'][target]).cpu()
        curr_acc = acc_func(curr_pred, curr_label, args, None)
        for acc_name in curr_acc:
            result[acc_name][target] = curr_acc[acc_name]
        
        curr_cfs = torch.cat(result['confounders'][target], 0)
        dcor1, dcor2 = calc_dcor(curr_label, curr_cfs, result, target, args)
        result['dcor1'][target] = dcor1
        result['dcor2'][target] = dcor2
        result['mean_dcor'][target] = (dcor1 + dcor2) / 2
        
        if test == True and target in args.confusion_matrix:
            calc_confusion_matrix(cfm, target, curr_pred, curr_label)
    
    result.pop('label')
    result.pop('pred')
    result.pop('embeddings')
    result.pop('confounders')
    
    return result
    
    
### ========= Train,Validate, and Test ========= ###
'''The process of calcuating loss and accuracy metrics is as follows.
   1) sequentially calculate loss and accuracy metrics of target labels with for loop.
   2) store the result information with dictionary type.
   3) return the dictionary, which form as {'cat_target':value, 'num_target:value}
   This process is intended to easily deal with loss values from each target labels.'''


'''All of the loss from predictions are summated and this loss value is used for backpropagation.'''    
# define training step
def train(net, trainloader, optimizer, scaler, args):
    net.train()
    train_result = set_result(args)
    update_interval = 1 if args.accumulation_steps is None else args.accumulation_steps
    
    for i, data in enumerate(trainloader,0):
        image = data[0]
        targets = [] if len(data) <= 1 else data[1]
        confounders = [] if len(data) <= 2 else data[2]
        image = image.to(f'cuda:{net.device_ids[0]}')
        
        cfs_batch = np.zeros((image.shape[0], 3))
        cfs_batch[:,0] = np.ndarray((0)) if targets == [] else targets['Autism'].float().detach().numpy()
        cfs_batch[:,1] = np.ndarray((0)) if confounders == [] else confounders.float().detach().numpy()
        cfs_batch[:,2] = np.ones((image.shape[0],))
        with torch.no_grad():
            net.module.cfs = nn.Parameter(torch.Tensor(cfs_batch).to('cuda:0'), requires_grad=False)
        
        with torch.cuda.amp.autocast():
            output = net(image)
            loss = calculating_loss_acc(targets, output, train_result, net, args)
            loss = loss / args.accumulation_steps if args.accumulation_steps is not None else loss 
        
        # output = net(image)
        # loss = calculating_loss_acc(targets, output, train_result, net, args)
        # loss = loss / args.accumulation_steps if args.accumulation_steps is not None else loss    
        # loss.backward() # when turn off amp.autocast
        scaler.scale(loss).backward()
        if ((i + 1) % update_interval == 0) or (i == (len(trainloader)-1)):
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True) # when turn off amp.autocast
            
        if confounders != []:
            train_result['confounders']['Autism'].append(confounders)

    # calculating total loss and acc of separate mini-batch
    train_result = total_loss_acc(train_result, args)
    
    return net, train_result


# define validation step
def validate(net, valloader, scheduler, args):
    val_result = set_result(args)

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            image = data[0]
            targets = [] if len(data) <= 1 else data[1]
            confounders = [] if len(data) <= 2 else data[2]
            image = image.to(f'cuda:{net.device_ids[0]}')
            
            cfs_batch = np.zeros((image.shape[0],3))
            cfs_batch[:,0] = np.ndarray((0)) if targets == [] else targets['Autism'].float().detach().numpy()
            cfs_batch[:,1] = np.ndarray((0)) if confounders == [] else confounders.float().detach().numpy()
            cfs_batch[:,2] = np.ones((image.shape[0],))
            net.module.cfs = nn.Parameter(torch.Tensor(cfs_batch).to('cuda:0'), requires_grad=False)
            
            with torch.cuda.amp.autocast():
                output = net(image)
                loss = calculating_loss_acc(targets, output, val_result, net, args)
                
            if confounders != []:
                val_result['confounders']['Autism'].append(confounders)
            
    val_result = total_loss_acc(val_result, args)
    
    # learning rate scheduler
    if scheduler:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(sum(val_result['auroc'].values()))
        else:
            scheduler.step()

    return val_result


def calc_confusion_matrix(confusion_matrices, curr_target, output, y_true):
    _, predicted = torch.max(output.data, 1)
    tn, fp, fn, tp = confusion_matrix(y_true.numpy(), predicted.numpy()).ravel()
    confusion_matrices[curr_target]['True Positive'] = int(tp)
    confusion_matrices[curr_target]['True Negative'] = int(tn)
    confusion_matrices[curr_target]['False Positive'] = int(fp)
    confusion_matrices[curr_target]['False Negative'] = int(fn) 
    
    
# define test step
def test(net, partition, args):
    def seed_worker(worker_id):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    testloader = torch.utils.data.DataLoader(partition['test'],
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
    
    test_result = set_result(args)
    confusion_matrices = defaultdict(defaultdict)

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            image = data[0]
            targets = [] if len(data) <= 1 else data[1]
            confounders = [] if len(data) <= 2 else data[2]
            image = image.to(f'cuda:{net.device_ids[0]}')
            
            cfs_batch = np.zeros((image.shape[0],3))
            cfs_batch[:,0] = np.ndarray((0)) if targets == [] else targets['Autism'].float().detach().numpy()
            cfs_batch[:,1] = np.ndarray((0)) if confounders == [] else confounders.float().detach().numpy()
            cfs_batch[:,2] = np.ones((image.shape[0],))
            net.module.cfs = nn.Parameter(torch.Tensor(cfs_batch).to('cuda:0'), requires_grad=False)
            
            output = net(image)
            
            if confounders != []:
                test_result['confounders']['Autism'].append(confounders)
            loss = calculating_loss_acc(targets, output, test_result, net, args)
            
    # caculating ACC and R2 at once  
    test_result = total_loss_acc(test_result, args, True, confusion_matrices)

    return test_result, confusion_matrices

## ============================================ ##
