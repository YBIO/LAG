
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# entropy calculation
def entropy(signal):
        lensig=signal.size
        symset=list(set(signal))
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
# read image
colorIm=Image.open('../37.png')
greyIm=colorIm.convert('L')
greyIm=np.array(greyIm)
N=3
S=greyIm.shape
E=np.array(greyIm)

#traverse image to calculate entropy
for row in range(S[0]):
    for col in range(S[1]):
        Left_x=np.max([0,col-N])
        Right_x=np.min([S[1],col+N])
        up_y=np.max([0,row-N])
        down_y=np.min([S[0],row+N])
        region=greyIm[up_y:down_y,Left_x:Right_x].flatten()  
        E[row,col]=entropy(region)

plt.subplot(1,3,1)
plt.imshow(colorIm)

plt.subplot(1,3,2)
plt.imshow(greyIm, cmap=plt.cm.gray)

plt.subplot(1,3,3)
plt.imshow(E, cmap=plt.cm.gray) #cmap=plt.cm.jet
# plt.colorbar()

plt.show()

