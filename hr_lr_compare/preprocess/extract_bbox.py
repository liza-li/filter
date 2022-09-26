
from extract_cnn_vgg16_keras import VGGNet
import cv2
import json
import jsonlines
import os
import numpy as np
import pickle
from keras.utils import load_img ,img_to_array
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

from keras.applications.vgg16 import VGG16
from numpy import linalg as LA
import h5py

 


if __name__ == "__main__":
    ann_file = '/home/zeli/projects/search/filter_coco.json'
    database = '/home/zeli/projects/datasets/coco/train2017'
    img_ids = '/home/zeli/projects/search/img_id.txt'
    index = '/home/zeli/projects/search/models/filter_coco_vgg_feature.h5'
 
        

    '''
   
    img_list = {}
    with open(img_ids, 'r') as f:
        data = f.readlines()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Remove the Enter at the end of each line
    for i, k in enumerate(data):
        split_data = k.strip('\n')
        img_list[i] = int(split_data)
    '''        
   # model = VGGNet()
    input_shape = (224, 224, 3)
    weight = 'imagenet'
    #weight = 'imagenet'
    pooling = 'max'
    model_vgg = VGG16(weights=weight,
                      input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                      pooling=pooling, include_top=False)
    
    json_list = []
    with open(ann_file, 'r') as f:
        for item in jsonlines.Reader(f):
            json_list.append(item)
    feats = [] 
    names = [] 
    count = 0      
    for i in json_list: 
        width = int(i[0]['width'])
        height = int(i[0]['height'])
        ratio_w = input_shape[0]/width
        ratio_h = input_shape[0]/height
        x =int(i[0]['bbox'][0]*ratio_w)
        y =int(i[0]['bbox'][1]*ratio_h)
        w =int(i[0]['bbox'][2]*ratio_w)
        h =int(i[0]['bbox'][3]*ratio_h)
    
       
        
        img_path = i[0]['image']
        img = load_img(img_path,target_size=(input_shape[0], input_shape[1]))
        img = img_to_array(img)
        
        mask = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
        mask[x:x+w,y:y+h,:]=1
        img = img*mask
        #img = img.resize(input_shape[0], input_shape[1],img.shape[2])
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = model_vgg.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((count + 1), len(json_list)))
        count=count+1
        
    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    # output = args["index"]
    output = index
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
       
        
        
        
            
        
   