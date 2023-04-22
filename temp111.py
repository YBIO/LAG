import os 
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from fightingcv_attention.attention.DANet import DAModule  # DANet 2019
from fightingcv_attention.attention.PSA import PSA # EPSANet 2021
from fightingcv_attention.attention.ECAAttention import ECAAttention # ECANet 2019

""" # a = torch.ones(3,3)
a = torch.randint(0, 22, (1,10,10))
print('a',a)
# b = torch.nonzero(a==1).squeeze()
# print(torch.masked_select(a, a==1))
index = torch.nonzero(a>10)
print('index:',index)

mask = torch.zeros(a.size())
for i in index:
    mask[i[0],i[1],i[2]] = 1
print('mask1:', mask)

# result = a * mask
# result1 = result[result>0]
result = torch.masked_select(a, mask)
result1 = result[result>0]

print('result:', result)
print('result1:', result1.size())
print(np.min(torch.Size([55]).item(),torch.Size([5]).item()))
 """

# def PCA_Batch_Feat(X, k, center=True):
#     """
#     param X: BxCxHxW
#     param k: scalar
#     return: k-dimension features
#     """
#     B, C, H, W = X.shape
#     X = X.permute(0, 2, 3, 1)  # BxHxWxC
#     X = X.reshape(B, H * W, C)
#     U, S, V = torch.pca_lowrank(X, center=center)
#     Y = torch.bmm(X, V[:, :, :k])
#     Y = Y.reshape(B, H, W, k) # BxHxWxk
#     Y = Y.permute(0, 3, 1, 2)  # BxkxHxW

#     return Y

# a = torch.rand(1, 256, 33, 33).type(torch.float32)
# psa = PSA(channel=256,reduction=8)
# danet=DAModule(d_model=256,kernel_size=3,H=33,W=33)
# # eca = ECAAttention(kernel_size=3)
# # Y = psa(a).type(torch.float16)
# Y = danet(a)

# print('dtype:', a.dtype, Y.dtype)

# print(Y.shape)

import torch
import torchvision.models as models

# from utils import loss as dcl

# resnet18 = models.resnet18()
# random_input = torch.rand((10, 3, 244, 244))
# print('input:',random_input.reshape(10, -1).size())   
# output = resnet18(random_input)
# print('output:', output.size())
# # for DCL
# loss_fn = dcl.DCL(temperature=0.5)
# loss = loss_fn(output, output)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

# # for DCLW
# loss_fn = dcl.DCLW(temperature=0.5, sigma=0.5)
# loss = loss_fn(output, output)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)



from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as mpatches
from skimage import data,filters,segmentation,measure,morphology,color

file1= 'sciuv_lev00_20230126T203543_exp_3000.04_Hg_fits'
file2 = 'sciuv_lev00_20230126T203543_exp_3000.04_Lg_fits'
file3 = 'sciuv_lev00_20230126T203547_exp_40000.04_Lg_fits'
file4 = 'sciuv_lev00_20230130T021542_exp_40000.04_Lg_fits'

hud = fits.open(file1)

img = hud[0].data
print(img.shape, type(img))
# 图像均值
mean_value = np.mean(img[1900:,1900:])
print('mean gray value:', mean_value)
print(img.shape, np.max(img), np.min(img))
# 阈值分割
min_value=10000
ret, thresh_img = cv2.threshold(img, min_value, 65535, cv2.THRESH_BINARY_INV)

# 形态学
# kernel = np.ones((7,7),np.uint8)
# erosion = cv2.erode(thresh_img, kernel, iterations = 1)
# thresh_img = cv2.dilate(erosion, kernel, iterations = 1) 

minus_img = img-thresh_img
ret1, minus_img_thres = cv2.threshold(minus_img, 800, 65535, cv2.THRESH_BINARY)

# kernel = np.ones((3,3),np.uint8)
# erosion = cv2.erode(minus_img_thres, kernel, iterations = 1)
# dilate = cv2.dilate(erosion, kernel, iterations = 1) 
# temp_img = cv2.addWeighted(minus_img, 0.5, minus_img_thres, 0.5, 0)
# lbimg=cv2.medianBlur(minus_img_thres, 5)

# 用背景像素填充
result_img=np.zeros(minus_img.shape)
for i in range(0, minus_img.shape[0]):        #行总数
    for j in range(0, minus_img.shape[1]):    #列总数
        value = minus_img[i, j]    #数组中的元素
        if value >=2000:
            result_img[i,j] = mean_value
        else:
            result_img[i,j] = minus_img[i, j] 



temp = cv2.Sobel(result_img, cv2.CV_16U, 1, 1)

#反相
for i in range(0, temp.shape[0]):        #行总数
    for j in range(0, temp.shape[1]):    #列总数
        temp[i,j]=65535-temp[i,j]

result_img = img-temp

plt.figure("Image")  
plt.subplot(151)
plt.imshow(img, cmap='gray')
plt.subplot(152)
plt.imshow(thresh_img, cmap='gray')
plt.subplot(153)
plt.imshow(minus_img, cmap='gray')
plt.subplot(154)
plt.imshow(temp, cmap='gray')
plt.subplot(155)
plt.imshow(result_img, cmap='gray')
# plt.axis('off') # 去坐标轴
# plt.xticks([]) # 去刻度
# plt.yticks([]) # 去刻度
# plt.savefig('file1.png', bbox_inches='tight',dpi=300,pad_inches=0.0)
plt.show()




