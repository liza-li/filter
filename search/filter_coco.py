

import os
import pickle
import cv2
import numpy as np
from pycocotools.coco import COCO
import json
import numpy
import jsonlines


def _xywh2cs(x, y, w, h):
    image_width = 96
    image_height = 128
    pixel_std = 200
    aspect_ratio = image_width * 1.0 / image_height
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)


def image_path_from_index(root, index):
    """ example: images / train2017 / 000000119993.jpg """
    file_name = '%012d.jpg' % index
    prefix = 'train2017'
    image_path = os.path.join(root, 'images', prefix, file_name)

    return image_path


def load_db(root):
    cocopath = f'{root}/annotations/person_keypoints_train2017.json'
    coco = COCO(cocopath)

    cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
    classes = ['__background__'] + cats
    num_classes = len(classes)
    _class_to_ind = dict(zip(classes, range(num_classes)))
    _class_to_ind = dict(zip(classes, range(num_classes)))
    _class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
    coco_ind_to_class_ind = dict(
                [
                    (_class_to_coco_ind[cls], _class_to_ind[cls])
                    for cls in classes[1:]
                ]
            )
    num_joints = 17
    image_set_index = coco.getImgIds()
    db = []
    for index in image_set_index:
        im_ann = coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = coco.loadAnns(annIds)
        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = coco_ind_to_class_ind[obj['category_id']]
            s = 128
            size_thr = s*s/4*3
            area = obj['bbox'][2] * obj['bbox'][3]
            if cls != 1:
                continue
            if obj['iscrowd']!=0:
                continue
            if area<size_thr:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=float)
            joints_3d_vis = np.zeros((num_joints, 3), dtype=float)
            for ipt in range(num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = _box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': image_path_from_index(root, index),
                'width':  width,
                'height': height,
                'bbox': obj['clean_bbox'],
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })
        db.append(rec)
    return db


def filter_box(db, area_thr):
    fdb = []
    for rec in db:
        frec = []
        for d in rec:
            area = d['bbox'][2] * d['bbox'][3]
            if area > area_thr:
                frec.append(d)
        if len(frec) > 0:
            fdb.append(frec)
    
    return fdb


def vis(db):
    for rec in db:
        for d in rec:
            bbox = d['bbox']
            joints = d['joints_3d']
            img_file = d['image']
            basename = os.path.basename(img_file)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            for j in range(17):
                cv2.circle(img, (int(joints[j][0]), int(joints[j][1])), 1, [0, 0, 255], 1)
            x,y,w,h = [int(i) for i in bbox]
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imwrite(basename, img)

def vis_(save_root, db):
    for rec in db:
        for d in rec:
            bbox = d['bbox']
            joints = d['joints_3d']
            img_file = d['image']
            basename = os.path.basename(img_file)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            for j in range(17):
                cv2.circle(img, (int(joints[j][0]), int(joints[j][1])), 1, [0, 0, 255], 1)
            # cimg = crop(img, bbox)
            x,y,w,h = [int(i) for i in bbox]
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imwrite(f'{save_root}/{basename}', img)

def crop(img, bbox):
    x,y,w,h = [int(i) for i in bbox]
    crop_img = img[y:y+h, x:x+w]
    return crop_img

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
            numpy.uint16,numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
            numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)): # add this line
            return obj.tolist() # add this line
        return json.JSONEncoder.default(self, obj)           


def create_db(root, sizes, fdb_prefix):
    db = load_db(root)
    # with open('db.pkl', 'wb') as fd:
    #     pickle.dump(db, fd)
    
    with open('/home/zeli/projects/search/filter_coco.json', 'a') as fd:
            for f in db:
                json.dump(f,fd)
    
    '''
    for s in sizes:
        fdb = filter_box(db, s*s/4*3)
        fdbname = f'{fdb_prefix}_fdb_{s}x{int(s/4*3)}.json'
        with open(fdbname, 'w') as fd:
            for f in fdb:
                json.dump(f,fd,cls=NpEncoder)
        fdb_names.append(fdbname)
    '''
    return fdb_names


def num_instance(db):
    num_instances = 0
    for rec in db:
        for d in rec:
            num_instances += 1
    return num_instances



if __name__ == '__main__':
    os.makedirs('fdb', exist_ok=True)
    # dataset_paths = {'coco':'coco', 'aic':'aic', 'hieve':'hieve/HIE20', 'posetrack': 'posetrack'}
    dataset_paths = {'coco':'coco'}
    sizes = [128]
    for dataset_name, dataset_path in dataset_paths.items():
        dataset_root = f'/datasets/{dataset_path}'
        fdb_prefix = f'fdb/{dataset_name}'
        db = load_db( dataset_root)
        with open('/home/zeli/projects/search/filter_coco.json', 'a') as fd:
            for f in db:
                json.dump(f,fd,cls=NpEncoder)
                fd.write('\n')
               # json.dump(",",fd)
                
        json_list = []
        with open('/home/zeli/projects/search/filter_coco.json', 'r') as ff:
            for item in jsonlines.Reader(ff):
                json_list.append(item)
        
        #fdb_names = create_db(dataset_root, sizes, fdb_prefix)
        '''
             for fdbname in fdb_names:
            
            with open(fdbname) as fd:
                fdb = fd.readlines()
                str = fdb[0].replace("][",",")
            with open('/home/zeli/projects/search/new_json.json', 'w') as fd:
                fd.write(json.dumps(str))    
            
                
            num = num_instance(fdb)
            print(fdbname, num)
    
        
        '''
   
    # save_root = 'vis'
    # os.makedirs(save_root, exist_ok=True)
    # vis(save_root, fdb)
