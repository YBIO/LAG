from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as mpatches
from skimage import data,filters,segmentation,measure,morphology,color

file1= 'sciuv_lev00_20230126T203543_exp_3000.04_Hg_fits'
file2 = 'sciuv_lev00_20230126T203543_exp_3000.04_Lg_fits'
file3 = 'sciuv_lev00_20230126T203547_exp_40000.04_Lg_fits'
file4 = 'sciuv_lev00_20230130T021542_exp_40000.04_Lg_fits'

hud = fits.open(file1)

image = hud[0].data

thresh =filters.threshold_otsu(image) #阈值分割

# bw =morphology.closing(image > thresh, morphology.square(3)) #闭运算
bw = morphology.opening(image > thresh, morphology.square(3)) #开运算
print('thresh:', thresh, np.min(bw), np.max(bw), np.mean(bw))
cleared = bw.copy()  #复制
# 形态学
# kernel = np.ones((7,7),np.uint8)
# erosion = cv2.erode(cleared, kernel, iterations = 1)
# cleared = cv2.dilate(erosion, kernel, iterations = 1) 

segmentation.clear_border(cleared)  #清除与边界相连的目标物

label_image =measure.label(cleared)  #连通区域标记
borders = np.logical_xor(bw, cleared) #异或
label_image[borders] = -1
image_label_overlay =color.label2rgb(label_image, image=image) #不同标记用不同颜色显示
result_img = image-bw

fig,(ax0,ax1,ax2,ax3)= plt.subplots(1,4, figsize=(8, 6))
ax0.imshow(image, plt.cm.gray)
ax1.imshow(bw, plt.cm.gray)
ax2.imshow(cleared,plt.cm.gray)
# ax3.imshow(image_label_overlay, plt.cm.gray)
ax3.imshow(result_img, plt.cm.gray)

# for region in measure.regionprops(label_image): #循环得到每一个连通区域属性集
    
#     #忽略大区域
#     if region.area > 2000:
#         continue

#     #绘制外包矩形
#     minr, minc, maxr, maxc = region.bbox
#     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                               fill=False, edgecolor='red', linewidth=2)
#     ax1.add_patch(rect)
fig.tight_layout()
plt.show()