import torch

import numpy as np
import pandas as pd
from itertools import product

from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from scipy.optimize import fminbound

### --- Calculate scores = d1 + beta dn + gamma dm ---
def calc_scores_given_hparams_vectorized(df, best_hparams, return_dn=False, torch_arr=False):
    if torch_arr:
        D_ns = torch.stack([torch.tensor(d) for d in df['D_n'].values])
        D_ms = torch.stack([torch.tensor(d) for d in df['D_m'].values])
        dists_tr_ns = torch.stack([torch.tensor(d) for d in df['dists_tr_n'].values])
        dists_tr_ms = torch.stack([torch.tensor(d) for d in df['dists_tr_m'].values])
        dists_ns = torch.stack([torch.tensor(d) for d in df['dists_n'].values])
        dists_ms = torch.stack([torch.tensor(d) for d in df['dists_m'].values])

        scaling_factors_n = torch.exp(-best_hparams['tau_1_n'] * D_ns) * torch.exp(-best_hparams['tau_2_n'] * dists_tr_ns)
        scaling_factors_m = torch.exp(-best_hparams['tau_1_m'] * D_ms) * torch.exp(-best_hparams['tau_2_m'] * dists_tr_ms)

        d_ns = torch.sum(scaling_factors_n * dists_ns, dim=1) / D_ns.shape[1]
        d_ms = torch.sum(scaling_factors_m * dists_ms, dim=1) / D_ms.shape[1]

        scores = torch.tensor(df['d_1'].values) + best_hparams['beta'] * d_ns + best_hparams['gamma'] * d_ms
    else:
        D_ns = np.stack(df['D_n'].values)
        D_ms = np.stack(df['D_m'].values)
        dists_tr_ns = np.stack(df['dists_tr_n'].values)
        dists_tr_ms = np.stack(df['dists_tr_m'].values)
        dists_ns = np.stack(df['dists_n'].values)
        dists_ms = np.stack(df['dists_m'].values)

        scaling_factors_n = np.exp(-best_hparams['tau_1_n'] * D_ns) * np.exp(-best_hparams['tau_2_n'] * dists_tr_ns)
        scaling_factors_m = np.exp(-best_hparams['tau_1_m'] * D_ms) * np.exp(-best_hparams['tau_2_m'] * dists_tr_ms)

        d_ns = np.sum(scaling_factors_n * dists_ns, axis=1) / D_ns.shape[1]
        d_ms = np.sum(scaling_factors_m * dists_ms, axis=1) / D_ms.shape[1]

        scores = df['d_1'].values + best_hparams['beta'] * d_ns + best_hparams['gamma'] * d_ms

    if return_dn:
        return scores, d_ns, d_ms
    else:
        return scores

def unpack_vector(x):
    return {
        'beta': x[0],
        'gamma': x[1],
        'tau_1_n': x[2],
        'tau_2_n': x[3],
        'tau_1_m': x[4],
        'tau_2_m': x[5],
    }

def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

### Calculate score and return negative of objective function (e.g., F1 score) for optimization
def optim_func(x, df, obj_func, obj_func_args):
    hparams = unpack_vector(x)
    y = df['is_mislabel'].values
    score = calc_scores_given_hparams_vectorized(df, hparams, return_dn = False)
    return -obj_func(y, score, **obj_func_args)

### Search for the best threshold to maximize F1 score
def optimize_f1_efficient(y, score, return_thres = False):
    def neg_f1(threshold):
        pred_label = score >= threshold
        return -f1_score(y, pred_label)
    best_thres = fminbound(neg_f1, score.min(), score.max(), xtol = 1e-8, disp = 0)
    best_f1 = -neg_f1(best_thres) 

    if return_thres:
        return best_f1, best_thres
    else:
        return best_f1

### Maximize over objective function (e.g., F1 score)
def maximize_metric(df, grid, obj_func, obj_func_args):
    best_x, best_val = None, -1

    for x in combinations_base(grid):
        g = []
        for i in ['beta', 'gamma', 'tau_1_n', 'tau_2_n', 'tau_1_m', 'tau_2_m']:
            if i in x:
                g.append(x[i])
            elif i in ['tau_1_n', 'tau_1_m']:
                g.append(x['tau_1'])
            elif i in ['tau_2_n', 'tau_2_m']:
                g.append(x['tau_2'])
            else:
                raise NotImplementedError(i)

        temp = optim_func(g, df, obj_func, obj_func_args)
        if -temp > best_val:
            best_val = -temp
            best_x = g

    score = calc_scores_given_hparams_vectorized(df, unpack_vector(best_x))
    return best_x, best_val, obj_func(df['is_mislabel'], score, return_thres=True, **obj_func_args)[1]


def eval_metrics(y, score, prevalence, fix_thres={}):
    if 'F1_optimal_thres' in fix_thres:
        f1_optim_thres = fix_thres['F1_optimal_thres']
    else:
        f1_optim, f1_optim_thres = optimize_f1_efficient(y, score, True)
        
    
    return {**prob_metrics(y, score), **{
        'F1_optimal_thres': f1_optim_thres
        },
        **binary_metrics(y, score >= f1_optim_thres, suffix = '_optimal'),
    }

def prob_metrics(targets, preds, sample_weight = None):
    return {
        'AUROC': roc_auc_score(targets, preds, sample_weight=sample_weight),
        'AUPRC': average_precision_score(targets, preds, average='macro', sample_weight=sample_weight)
    }

def binary_metrics(targets, preds, label_set=[0, 1], suffix='', return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'F1': f1_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return {f"{i}{suffix}": res[i] for i in res}