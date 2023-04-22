import os 
import torch
import torchvision
import numy as np


def cal_unvertainty_pseudo_label(pred_prob: torch.tensor, prev_num_classes, unknown=True):


    # initial parameters
    uncertainty_score_list = []
    for i in (1, prev_num_classes):
        uncertainty_score[i] = 


