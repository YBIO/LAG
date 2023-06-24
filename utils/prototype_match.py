'''
 * @Author: YBIO
 * @Date: 2022-11-14 09:44:28 
 * @Last Modified by:   YBIO 
 * @Last Modified time: 2022-11-14 09:44:28 
 '''

import os
import numpy as np
import torchvision
import cv2
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm


from fightingcv_attention.attention.DANet import DAModule  # DANet 2019
from fightingcv_attention.attention.PSA import PSA # EPSANet 2021
from fightingcv_attention.attention.ECAAttention import ECAAttention # ECANet 2019


def PCA_Batch_Feat(X, k, center=True):
    """
    param X: BxCxHxW
    param k: scalar
    return: k-dimension features
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U, S, V = torch.pca_lowrank(X, center=center)
    Y = torch.bmm(X, V[:, :, :k])
    Y = Y.reshape(B, H, W, k) # BxHxWxk
    Y = Y.permute(0, 3, 1, 2)  # BxkxHxW

    return Y


class proto_computing:
    def  __init__(self, args):
        self.args = args
        self.curr_step = opts.curr_step
        self.num_classes = opts.num_classes
        self.task = opts.task

    def protosave(outputs: torch.tensor, ret_features: torch.tensor, outputs_prev: torch.tensor, ret_features_prev: torch.tensor, min_classes=10, use_sigmoid=True, unknown=True):
<<<<<<< HEAD
        """ 
=======
        I""" 
>>>>>>> ecae318e6dc743ca5dadc27edce5539b03438991
        Input:
            outputs: current model output
            ret_features: current model layer-wise features
            outputs_prev: previous model output
            ret_features_prev: previous model layer-wise features
        Output:
            SS_prototype : prototype for sample-specific features
            SI_prototype: prototype for semantic-invariant features
        """
        ## calculate fearure mean
        curr_num_classes = int(self.task.split('-')[0]) + self.curr_step * int(self.task.split('-')[-1])
        batch_features_mean = torch.zeros([curr_num_classes, 2048], device=self.device)

        # Distangled Representation Learning for sample-specific and semantic-invariant features 
        # for sample-specific (SS) features
        SS_prototype= torch.zeros([curr_num_classes, 2048]) 
        # for semantic-invariant (SI) features
        SI_prototype = torch.zeros([curr_num_classes, 2048])
        
        SS_prototype = ret_features['feature_out'].detach()
        SI_prototype = torch.sum(ret_features['feature_l1'].detach()) 
        
 


        # calculate feature mean
        # prev_feature_mean: feature mean of previous model
        # local_feature_mean: feature mean of current batch     
            
        return prototypes, count_features

    def prototype_matching():
        
        for cl in curr_num_classes:
            batch_feature_mean = torch.mean(ret_features['feature_out'], dim=-1)
            torch.max(ret_features['features_out'],dim=-1)
            
            
        for cl in curr_num_classes:
            features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[1],-1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl
            prototype_features = features.append(prototype_cl)