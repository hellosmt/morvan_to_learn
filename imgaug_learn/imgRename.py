import os

path_name='/media/sunmengtuo/UBUNTU 16_0/garbage-城管/垃圾桶'
#path_name :表示你需要批量改的文件夹
i=0
for item in os.listdir(path_name):#进入到文件夹内，对每个文件进行循环遍历
    os.rename(os.path.join(path_name,item),os.path.join(path_name,('Img'+str(i)+'.jpg')))#os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
    i+=1
