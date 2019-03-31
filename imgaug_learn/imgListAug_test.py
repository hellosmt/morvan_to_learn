# coding:utf-8

# 对文件夹里的所有图片进行增强，没有考虑bounding box
# 整体流程为：定义变换序列（Sequential）→读入图片（imread）→执行变换（augment_images）→保存图片（imwrite）
from imgaug import augmenters as iaa
import cv2
import os
import numpy as np

imgList = []

seq = iaa.Sequential([
    iaa.Flipud(0.5),  # 镜面翻转
    iaa.Fliplr(0.5),  # 水平翻转 0.5是随机选一半的图片进行翻转
    iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯模糊 其实我不懂这个
    iaa.Affine(
        translate_px={"x": (-15, 15), "y": (-15, 15)},  # 在x，y轴上偏移（-15， 15）个像素点，两个轴上的偏移是独立的
        scale=(0.8, 1.2),  # 缩放0.8-1.2的倍数，x y的比例是不变的，和上面的x y独立的偏移是不同的
        rotate=(-45, 45)  # 旋转
               )
])

for filename in os.listdir("imgs/"):
    if filename.endswith('.jpg'):  # 文件是.jpg后缀的才进行处理
        img = cv2.imread(os.path.join("imgs/", filename))
        # img = np.array(img)   # 这一步一开始没写报错，后面注释掉不报错，不知道为什么？
        # 查了一下，如果是用PIL.Image.open()打开一个图片，要想使用img.shape函数，就要先将image格式转成array格式
        imgList.append(img)

k = 1000  # 新图片的名字起始数量
AUG_LOOP = 5   # 每张图片增强的数量

for i in imgList:  # 原图片放在了imgList里面
    for epoch in range(AUG_LOOP):
        imgAug = seq.augment_image(i)  # 对原图片进行增强
        cv2.imwrite("imgsAug/"+str(k)+"aug_"+str(epoch)+".jpg", imgAug)
    k = k+1
