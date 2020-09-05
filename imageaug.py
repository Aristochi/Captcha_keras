# import os
#
# from PIL import Image
# import numpy as np
# path='./train/'
# imgpath=os.listdir('./train/')
# savepath='./traingray/'
# for imgs in imgpath:
#     image=path+str(imgs)
#     img = Image.open(image)
#     img = img.convert('L') # 图像二值化
#     img.save(savepath+str(imgs))
import os

import cv2
from imgaug import augmenters as iaa, ia  # 引入数据增强的包

sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，
path='./train/'
imglist=os.listdir('./train/')
savepath='./trainaug/'
seq = iaa.Sequential(
    [
        iaa.SomeOf((1,3),[
        iaa.Multiply(mul=1.05, per_channel=False, name=None, deterministic=False, random_state=None),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.ContrastNormalization((0.75, 1.5), per_channel=0.5)
        ])

    ],
    random_order=True  # 随机的顺序把这些操作用在图像上
)
# 批量图像增强，每张图像增强5次

for i in imglist:
    imgs=path+str(i)
    img_aug=cv2.imread(imgs)
    images_aug = seq.augment_images(img_aug)  # 实现图像增强
    cv2.imwrite(savepath+str(i), images_aug)
