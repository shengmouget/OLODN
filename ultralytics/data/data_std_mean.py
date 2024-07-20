import numpy as np
import cv2
import os 

img_h,img_w = 128,128
means,stdevs = [],[]
img_list = []

imgs_path = "/home/neuedu/桌面/yolov5_yuan/dataset/images/"
imgs_path_list = os.listdir(imgs_path)
len_ = len(imgs_path_list)
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:,:,:,np.newaxis]
    img_list.append(img)
imgs = np.concatenate(img_list,axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:,:,i,:].ravel() # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
# RGB < --  BGR
means.reverse()
stdevs.reverse()
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))