from itertools import combinations
# import pdb

import dcor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_fscore_support

# NTXentLoss removed (not used in paper)


def contrastive_loss(output, result, metric='cos'):
    loss_name = 'contrastive_loss_'+metric
    loss_pos = 0
    loss_neg = 0
    
    out_idxs = list(combinations(range(output.shape[1]), 2))
    # calc contrastive loss between two modalities from (Num_modalities(M) combinations 2, M_C_2). In case of more than 2 modalities
    for out_idx in out_idxs:
        embedding_1, embedding_2 = output[:,out_idx[0]], output[:,out_idx[1]]
        if metric == 'cos':
            embedding_2_rolled = embedding_2.roll(1,0)           
            criterion_ssim = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
            label_positive = torch.ones(embedding_1.shape[0], device='cuda:0')
            label_negative = -torch.ones(embedding_1.shape[0], device='cuda:0')
            loss_pos += criterion_ssim(embedding_1, embedding_2, label_positive)
            loss_neg += criterion_ssim(embedding_1, embedding_2_rolled, label_negative)
            
        elif metric.upper() == 'L2':
            criterion_ssim = nn.MSELoss()
            loss_pos += criterion_ssim(embedding_1, embedding_2)
        else:
            raise Exception(f"ERROR: Invalid metric was used!! <{metric}>")
            
    loss = (loss_pos + loss_neg)/2
    if metric == 'cos':
        result['loss'][f'{loss_name}_positive'].append(loss_pos.item())
        result['loss'][f'{loss_name}_negative'].append(loss_neg.item())
    else:
        result['loss'][loss_name].append(loss_pos.item())
        
    return loss
    
    
def calc_acc(tmp_output, label, args, tmp_loss=None):
    _, predicted = torch.max(tmp_output.data, 1)
    correct = (predicted == label).sum().item()
    total = label.size(0)
    acc = (100 * correct / total)

    return {'acc': acc}


def calc_acc_auroc_ap(tmp_output, label, args, tmp_loss=None, **kwargs):
    _, pred = torch.max(tmp_output, 1)
    prob = F.softmax(tmp_output, 1)[:,1]
    if torch.any(torch.isnan(prob)):
        print("WARNING: prob contains NaN")
        prob = torch.nan_to_num(prob)
    correct = (pred == label).sum().item()
    total = label.size(0)   
    
    label, prob, pred = label.cpu(), prob.cpu(), pred.cpu()
    acc = (100 * correct / total)   
    auroc = roc_auc_score(label, prob).item()
    ap = average_precision_score(label, prob).item()
    
    pc, rc, f1, sup = precision_recall_fscore_support(label, pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    npv = tn/(tn+fn)
        
    result = {'acc': acc, 'auroc': auroc, 'ap': ap,
              'precision':pc, 'recall':rc, 'f1':f1, 'npv':npv}
    for k, v in result.items():
        result[k] = round(v,4)
    
    return result


def calc_R2(tmp_output, y_true, args, tmp_loss=None, **kwargs): #230313change
    if ('MAE' in args.exp_name or tmp_loss == None):
        criterion = nn.MSELoss()
        tmp_loss = criterion(tmp_output.float(), y_true.float().unsqueeze(1))

    y_var = torch.var(y_true, unbiased=False)
    r_square = 1 - (tmp_loss / y_var)
                    
    return {'r_square': r_square.item()}


def calc_MAE_MSE_R2(tmp_output, y_true, args, tmp_loss=None):
    pred, true = tmp_output.float(), y_true.float().unsqueeze(1)
    abs_loss = torch.nn.functional.l1_loss(pred, true)
    mse_loss = torch.nn.functional.mse_loss(pred, true)
    y_var = torch.var(true, unbiased=False)
    r_square = 1 - (mse_loss / y_var)
    result = {
        'abs_loss': abs_loss.item(),
        'mse_loss': mse_loss.item(),
        'r_square': r_square.item()
    }
    
    return result


def calc_dcor(labels, cfs, result, target, args):
    features = torch.cat(result['embeddings'][target], 0) # Iterations x Batch x Modaliteis x Hidden_dim => Num_data(IxB) x M x H
    features = features.reshape(features.shape[0],-1).numpy() # Num_data x Concatenated_feature(MH)

    i0 = np.where(labels == 0)[0]
    i1 = np.where(labels == 1)[0]
    dc0 = dcor.u_distance_correlation_sqr(features[i0], cfs[i0])
    dc1 = dcor.u_distance_correlation_sqr(features[i1], cfs[i1])
    
    step = 'train' if len(labels) > 90 else 'valid'
    step = 'test' if len(i0) == len(i1) else step

    print(f"{step} dcorr for class 0 & 1: {dc0:.4f} / {dc1:.4f}")
    
    return dc0, dc1

    
def calculating_loss_acc(targets, output, result, net, args, test=False):
    '''define calculating loss and accuracy function used during training and validation step'''
    # << should be implemented later >> how to set ssim_weight?
    if targets != []:
        cat_weight = (len(args.cat_target)/(len(args.cat_target)+len(args.num_target)))
        num_weight = 1 - cat_weight
    loss = 0.0
    
    # calculate constrastive_loss
    if args.metric and (len(args.data_type) > 1 and args.in_channels == 1):
        loss = contrastive_loss(output['embeddings'].cuda(), result, args.metric)
        
    # calculate target_losses & accuracies
    for curr_target in args.cat_target+args.num_target:
        tmp_output = output[curr_target]
        label = targets[curr_target].to('cuda:0')
        tmp_label = label.long() if curr_target in args.cat_target else label.float().unsqueeze(1)
        weight = cat_weight if curr_target in args.cat_target else num_weight
        weight = -0.1*weight if curr_target in args.use_negative_loss else weight
        if curr_target in args.cat_target:
            class_weight = None if curr_target not in args.use_weighted_loss else torch.Tensor(args.class_weights[curr_target]).to('cuda:0')
            label_smoothing = args.use_label_smoothing if (0<=args.use_label_smoothing<=1) else 0
            criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
        elif 'MAE' in args.exp_name: #230313change
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        
        # Loss
        tmp_loss = criterion(tmp_output.float(), tmp_label)
        loss += tmp_loss * weight
        result['loss'][curr_target].append(tmp_loss.item())
        result['label'][curr_target].append(label) 
        result['pred'][curr_target].append(tmp_output.detach().float()) 
        result['embeddings'][curr_target].append(output['embeddings'])
        
    if torch.any(torch.isnan(loss)):
        print("loss nan")
        loss = torch.nan_to_num(loss)
                
    return loss
