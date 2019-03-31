# 查看哪些图片不是RGB模式的

from PIL import Image
import os

IMG_DIR = "C:\\Users\\sunmengtuo\\Desktop\\program\\城市管理项目\\garbage-img\\"

for filename in os.listdir(IMG_DIR):  # os.listdir()列出文件夹下的所有
    if filename.endswith(".jpg"):  # 如果文件是图片
        img = Image.open(os.path.join(IMG_DIR, filename))
        if img.mode != "RGB":
            print(filename, img.mode)