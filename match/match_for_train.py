# match for train
# inpute >> HR
    # Affine: HR-LR-HR'  (to make easier, we use same database/json)
    # Extract: 
        # Support Set: bbox feature, and save as h5 file
        # Query Set: image feature   
    # Match: match and save list in json


from inspect import CORO_CLOSED
from re import X
import cv2
import numpy as np

import jsonlines
from PIL import Image

from keras.utils import load_img ,img_to_array
#from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
#from keras.applications.vgg16 import VGG16
from numpy import linalg as LA
import h5py
from extract_cnn_vgg16_keras import VGGNet
import os
import torchvision.transforms as transforms
#img_to_tensor = transforms.ToTensor()

target_size = [224,224]


def get_db(ann_file):
    json_list = []
    with open(ann_file, 'r') as f:
        for item in jsonlines.Reader(f):
            json_list.append(item)
    results = []
    result = []
    for i in json_list: 
        width = i[0]['width']
        height = i[0]['height']
        image_size = [width,height]
        imgpath = i[0]['image']
        img = Image.open(imgpath)
        center = i[0]['center']
        scale = i[0]['scale'] 
        bbox = i[0]['bbox'] 
        
        results.append(
            {
                'image_size' :image_size,
                'imgpath':imgpath,
                'img':img,
                'center':center,
                'scale':scale,
                'rotation':0, 
                'bbox':bbox
            }  
        )
    
    
    return results

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    
    shift = np.array(shift)
    center = np.array(center)
    scale = np.array(scale)
    

    # pixel_std is 200.
    scale_tmp = scale * 200.0


    
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans        

def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt

def affine(results):
    
   img_list = [] 
   for j in results: 
       img_lr = []
       image_size = j['image_size']
       imgpath = j['imgpath']
       img = j['img']
       c = j['center']
       s = j['scale']
       r = j['rotation']
       bbox = j['bbox'] 
       img = img_to_array(img)
       x =int(j['bbox'][0])
       y =int(j['bbox'][1])
       w =int(j['bbox'][2])
       h =int(j['bbox'][3])
       mask = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
       mask[y:y+h,x:x+w,:]=1
       img = img*mask
       cv2.imwrite("/home/zeli/projects/match/hr.jpg",img)
       
       

       
       trans = get_affine_transform(c, s, r, LR_size)
       img = cv2.warpAffine( img,trans, (int(LR_size[0]), int(LR_size[1])),flags=cv2.INTER_LINEAR)
       
       #x_l = j['bbox'][0]
       #y_l = j['bbox'][1]
       #[x_l,y_l] = affine_transform([x_l,y_l] , trans)
       #x_r = j['bbox'][0]+j['bbox'][2]
       #y_r = j['bbox'][1]+j['bbox'][3]
       #[x_r,y_r] = affine_transform([x_r,y_r] , trans)
       #w = x_r-x_l
       #h = y_r-y_l
       
       img_x=c[0]-s[0]*1/2
       img_y=c[1]-s[1]*1/2
       img_coord = [img_x,img_y]
       
       c=affine_transform(c, trans)
       img_coord=affine_transform(img_coord, trans)
       s[0]=(c[0]-img_coord[0])*2
       s[1]=(c[1]-img_coord[1])*2
       
      #s=affine_transform(s, trans)
       
       cv2.imwrite("/home/zeli/projects/match/lr.jpg",img)
       
       #img2 = cv2.resize(img,(64,64),interpolation=cv2.INTER_LINEAR)
       #cv2.imwrite("/home/zeli/projects/match/lr.jpg",img2)
       
       
       #img2 = cv2.resize(img,(224,224),interpolation=cv2.INTER_NEAREST)
       #cv2.imwrite("/home/zeli/projects/match/lr22.jpg",img2)
       
       trans = get_affine_transform(c, s, r, target_size)
       img = cv2.warpAffine( img,trans, (int(target_size[0]), int(target_size[1])),flags=cv2.INTER_LINEAR)
       
       #[x_l,y_l] = affine_transform([x_l,y_l] , trans)
       #if y_l<0:
        #   y_l=0
       #if x_l<0:
        #   x_l=0
       #[x_r,y_r] = affine_transform([x_r,y_r] , trans)
       #w = x_r-x_l
       #h = y_r-y_l
       cv2.imwrite("/home/zeli/projects/match/lhr.jpg",img)
       
      
       #bbox[0] =x_l
       #bbox[1] =y_l
       #bbox[2] =w
       #bbox[3] =h
       
       img_list.append(
           {
               'imgpath':imgpath,
               'img':img,
               'image_size':image_size,
               #'bbox':bbox
           }
       )
       
       
   return img_list

def extract_match(img_list):
    match_list = []
    for k in img_list: 
        match = []
        imgpath = k['imgpath']
        img_id =  os.path.split(imgpath)[1]
        img = k['img']
        #x =int(k['bbox'][0])
        #y =int(k['bbox'][1])
        #w =int(k['bbox'][2])
        #h =int(k['bbox'][3])
        #mask = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
        #mask[x:x+w,y:y+h,:]=1
        #img = img*mask
        #cv2.imwrite("/home/zeli/projects/match/mask.jpg",img)
        #img = img.resize(input_shape[0], input_shape[1],img.shape[2])
        #img = np.expand_dims(img, axis=0)
        #img = preprocess_input_vgg(img)
        feat = model.vgg_extract_feat_bbox(img) 
        #norm_feat = feat[0] / LA.norm(feat[0])
        scores = np.dot(feat, feats.T)
        rank_ID = np.argsort(scores)[::-1]
        #rank_score = scores[rank_ID]
        for i, index in enumerate(rank_ID[0:maxres]):
            match.append(imgNames[index])
        match_list.append(
            {
                'source_img':img_id,
                'match_imgs':match
                
            }
                          )
    return match_list



if __name__=="__main__":
   
    ann_file = '/home/zeli/projects/match/test.json'
    database = '/home/zeli/projects/datasets/coco/train2017'
    index = '/home/zeli/projects/search/models/filter_coco_vgg_feature.h5'
    input_shape = (224, 224, 3)
    LR_size = [48,64] 
    weight = 'imagenet'
    pooling = 'max'
    model = VGGNet()
    maxres = 3

    h5f = h5py.File(index, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()
    results = get_db(ann_file)
    img_list = affine(results)
    match_list = extract_match(img_list)
    for item in match_list:
        print(item)    
    
       
    
        
        
        
       
        
    

       
    
       
      
            
    
    





