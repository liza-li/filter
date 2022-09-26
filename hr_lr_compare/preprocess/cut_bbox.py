import os
import shutil
from os.path import join
import cv2
import glob
import jsonlines

root_dir = "/home/zeli/projects/datasets/coco/images/train2017" #原始图片保存的位置
save_dir = "/home/zeli/projects/search/database" #生成截取图片的位置
json_dir = "/home/zeli/projects/search/filter_coco.json"



max_s = -1
min_s = 1000


json_list = []
with open(json_dir, 'r') as f:
    for item in jsonlines.Reader(f):
        json_list.append(item)

num = 0 #避免同一图像多个实例存储冲突 
for i in json_list: 
    img_path = i[0]['image']
    img_name = os.path.split(img_path)[1]
    img = cv2.imread(img_path)
    height, width, channel = img.shape #得到图片的尺寸
    x =i[0]['bbox'][0]
    y =i[0]['bbox'][1]
    w =i[0]['bbox'][2]
    h =i[0]['bbox'][3]
    
    max_s = max(w*h, max_s)
    min_s = min(w*h, min_s)
    
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    crop_img = img[y1:y2, x1:x2]
    
    new_jpg_name = img_name.split('.')[0] + "_crop_" + str(num) + ".jpg" #存储图片的名称
    print("已截取",num+1,"张图片")
    num=num+1
    cv2.imwrite(os.path.join(save_dir, new_jpg_name), crop_img) #截取的图片
    
    
    
    
    
    
    
    
    
        
