import cv2
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

def getFileList(dir,Filelist, ext=None):

    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist

label_path = "/home/yb/dataset/ISPRS_2D/postdam/GT_final_wcx_600/"

labellist = getFileList(label_path, [], 'png')
labellist = sorted(labellist)
print('Acquiring '+str(len(labellist))+' labels\n')

## store class index for each image

result_file = open('result.txt', 'w', encoding='utf-8') 
for item in tqdm(range(0,len(labellist))):

    gray_list = []
    label = Image.open(labellist[item])
    label = np.asarray(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j] not in gray_list:
                gray_list.append(label[i, j])
    gray_list.sort()
    
    result_file.write(labellist[item].split('/')[-1].split('.')[0])
    for i in gray_list:
        result_file.write(' ' + str(i))
    result_file.write('\n')
    
