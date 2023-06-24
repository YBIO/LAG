""" 
 * @Author: YBIO 
 * @Date: 2022-12-27 19:35:00 
 """
import os
 import torch
 import torchvision
 import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import cv2
import random

from PIL import Image
from tqdm import tqdm
from utils.loss import CircleLoss, convert_label_to_similarity, ContrastiveLoss, DCL, DCLW

def cal_entropy(ret_feat):
    entropy = ret_feat * (np.log(ret_feat))

    return entropy

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def  cal_similarity(outputs: torch.tensor, ret_features: torch.tensor, outputs_prev: torch.tensor, ret_features_prev: torch.tensor, rho=0.5, use_sigmoid=True, unknown=True):
    """ 
    Args:
        outputs: model output map
        ret_features: features from deep layers
    Return:
        SI_feats, SS_feats: features ranking by entropy measure
     """

    if use_sigmoid:
        pred_prob = torch.sigmoid(outputs).detach()
        pred_prob_prev = torch.sigmoid(outputs_prev).detach()
    else: 
        pred_prob = torch.softmax(outputs, 1).detach()
        pred_prob_prev = torch.softmax(outputs_prev, 1).detach()

    pred_scores, pred_labels = torch.max(pred_prob, dim=1)  # [b,1,513,513]
    pred_scores_prev, pred_labels_prev = torch.max(pred_prob_prev, dim=1)  # [b,1,513,513]

    channel_dim = ret_features['feature_out'].size()[1]
    channel_dim_prev = ret_features_prev['feature_out'].size()[1]

    assert channel_dim == channel_dim_prev

    opt_loss = torch.tensor(0.)
    rank_dict = {}

    for dim in channel_dim:
        ret_entropy_index = cal_entropy(ret_features['feature_out'][:,dim,:,:])
        ret_entropy_index_prev = cal_entropy_prev(ret_features['feature_out'][:,dim,;,:])

        dis_en = KLDiv_loss(ret_entropy_index, ret_entropy_index_prev)
        rank_dict[dim] = dis_en

    sort_rank_dict = sorted(rank_dict.items(), key=lambda x: x[1])

    num_SI_feat = rho * len(sort_rank_dict)
    num_SS_feat = (1.-rho) * len(sort_rank_dict)
    SI_feat_dict = dict_slice(sort_rank_dict, 0, num_SI_feat)
    SS_feat_dict = dict_slice(sort_rank_dict, num_SI_feat+1, len(sort_rank_dict))

    
    SI_channel_index_list = SI_feat_dict.keys()
    SS_channel_index_list = SS_feat_dict.keys()

    return SI_channel_index_list, SS_channel_index_list
    


