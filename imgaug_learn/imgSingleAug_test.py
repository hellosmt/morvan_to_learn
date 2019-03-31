# coding:utf-8

#对单张图片处理
# 整体流程为：定义变换序列（Sequential）→读入图片（imread）→执行变换（augment_images）→保存图片（imwrite）
from imgaug import augmenters as iaa
import cv2

seq = iaa.Sequential([
    iaa.Fliplr(0.5),   # 水平翻转 0.5是随机选一半的图片进行翻转
    iaa.Flipud(0.5)    # 镜面翻转
])

img = cv2.imread("imgSingle/test.jpg")
img_aug = seq.augment_image(img)  # 对单个图片进行增强处理
cv2.imwrite("imgSingle/imgaug1.jpg", img_aug)