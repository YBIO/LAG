#!/usr/bin/env python3

import typing
from collections import defaultdict
from typing import Any, cast, List, Tuple, Union

import torch.nn as nn
import torchvision
from torchvision import models
from captum._utils.common import (
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _register_backward_hook,
    _run_forward,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import Literal, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.common import _sum_rows
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from captum.attr import LRP
from captum.attr import visualization as viz
import numpy as np
from PIL import Image




if  __name__ == '__main__':
    import sys
    import cv2
    import torch
    import torch.nn as nn
    sys.path.append("..")
    import network
    import utils
    
    ## read data
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    img1 = cv2.imread('../2007_000364.jpg')
    # img2 = cv2.imread('../2007_000027.jpg')
    img2 = cv2.imread('../00001.png')
    img1 = cv2.resize(img1, (500,500))
    img2 = cv2.resize(img2, (500,500))
    img_tensor_1 = torch.from_numpy(img1)
    img_tensor_2 = torch.from_numpy(img2)
    img_tensor_1 = img_tensor_1.to(device, dtype=torch.float32, non_blocking=True)
    img_tensor_2 = img_tensor_2.to(device, dtype=torch.float32, non_blocking=True)
    print('img_tensor_before:', img_tensor_1.size())
    img_tensor_1 = img_tensor_1.permute(2,0,1)
    img_tensor_2 = img_tensor_2.permute(2,0,1)
    print('img_tensor_after:', img_tensor_1.size())
    input_tensor = torch.stack((img_tensor_1, img_tensor_2),0) 
    # img = torch.rand(1, 3, 33, 33)#.type(torch.float32)
        


    ## load model
    """ model = network.deeplabv3_resnet101()
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    checkpoint = '../checkpoints/deeplabv3_resnet101_voc_15-1_step_0_overlap.pth'
    model.load_state_dict(checkpoint, strict=False)
    model = nn.DataParallel(model) 
    model = model.to(device) """
    model = models.resnet101(pretrained=False)
    # checkpoint_path = '../checkpoints/deeplabv3_resnet101_voc_15-1_step_0_overlap.pth'
    checkpoint_path = '/home/yb/code/semantic/SSUL/checkpoints/deeplabv3_resnet101_ISPRS_2-2-1_step_0_overlap.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_state"]
    print('check_checkpoint:', type(checkpoint))
    model.load_state_dict(checkpoint, strict=False)
    model = model.eval()

    ## LRP
    LRP_model = LRP(model)
    attribution = LRP_model.attribute(input_tensor, target=2)  #目标类别是5
    print('attribution:', type(attribution), attribution.size())
    attri_img1, attri_img2 = attribution.split(1, 0) #将一个batch的tensor分割为两张图的tensor
    attri_img1 = attri_img1.reshape(3,500,500) #去掉第一个维度
    attri_img2 = attri_img2.reshape(3,500,500) #去掉第一个维度
    print('attri_img1:', attri_img1.size())
    print('attri_img2:', attri_img2.size())
    save_img1 = ((attri_img1.cpu().detach().numpy().transpose(1,2,0)+1) / 2) 
    save_img1 = save_img1 * 255.0
    save_img1 = np.array(save_img1, dtype='uint8')
    save_img1= cv2.cvtColor(save_img1, cv2.COLOR_BGR2RGB)
    print('save_fig1:', type(save_img1), save_img1.shape)
  
    # cv2.imwrite('attri_img1.jpg', save_img1)  

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.3, '#000000'),
                                                  (1, '#000000')], N=256)



    # cmap = plt.get_cmap("turbo_r")
    # default_cmap = LinearSegmentedColormap.from_list("trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=0.0, b=1.0), 
    #                                                                                         cmap(np.linspace(0.0, 1.0, 100)),)


    _ = viz.visualize_image_attr(np.transpose(img_tensor_1.squeeze().cpu().detach().numpy(),(1,2,0)),
                                np.transpose(attri_img1.squeeze().cpu().detach().numpy(),(1,2,0)),
                                method='heat_map',
                                cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)
    _ = viz.visualize_image_attr(np.transpose(img_tensor_2.squeeze().cpu().detach().numpy(),(1,2,0)),
                                np.transpose(attri_img2.squeeze().cpu().detach().numpy(),(1,2,0)),
                                method="heat_map",
                                cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)
