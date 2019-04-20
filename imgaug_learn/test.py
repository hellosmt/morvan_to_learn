import os
import os.path
import shutil

IMG_DIR = "/media/sunmengtuo/UBUNTU 16_0/garbage-城管/garbage bag"
XML_DIR = "/media/sunmengtuo/UBUNTU 16_0/garbage-城管/garbage-xml"

xml_name_list = []
for filename in os.listdir(IMG_DIR):
    if filename.endswith(".jpg"):
        #print(filename)
        xml_name_list.append(filename[:-4]+'.xml')

for filename in os.listdir(XML_DIR):
    if filename in xml_name_list:
        #print(filename)
        sourceFile = os.path.join(XML_DIR, filename)
        shutil.copy(sourceFile, IMG_DIR)