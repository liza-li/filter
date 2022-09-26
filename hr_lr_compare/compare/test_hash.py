
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import json
from compare_test import aHash,dHash,pHash,cmpHash

query = '/home/zeli/projects/search/9.jpg'
index = '/home/zeli/projects/search/models/hash_database_feature.json'
result = '/home/zeli/projects/search/database'

queryImg = mpimg.imread(query)
qahash =aHash(queryImg)
qdhash =dHash(queryImg)
qphash =pHash(queryImg)

with open(index,'r') as load_f:
    load_dict = json.load(load_f)
#print(load_dict)
hashn1 = []
hashn2 = []
hashn3 = []
names = []
for hash in load_dict:
   # print(i[0]['aHash'])
    #print(type(i))
    ahash= hash[0]['aHash']
    dhash= hash[0]['dHash']
    phash= hash[0]['pHash']
    name= hash[0]['name']
    n1=cmpHash(qahash,ahash)
    n2=cmpHash(qdhash,dhash)
    n3=cmpHash(qphash,phash)
    hashn1.append(n1)
    hashn2.append(n2)
    hashn3.append(n3)
    names.append(name)
#print(type(hashn1))

hashn1 = np.asarray(hashn1)
hashn2 = np.asarray(hashn2)
hashn3 = np.asarray(hashn3)

names = np.asarray(names)

rank_ID_n1 = np.argsort(hashn1)[::-1]
rank_ID_n2 = np.argsort(hashn2)[::-1]
rank_ID_n3 = np.argsort(hashn3)[::-1]

rank_score_n1 = hashn1[rank_ID_n1]
rank_score_n2 = hashn2[rank_ID_n2]
rank_score_n3 = hashn3[rank_ID_n3]

rank_name_n1 = names[rank_ID_n1]
rank_name_n2 = names[rank_ID_n2]
rank_name_n3 = names[rank_ID_n3]




maxres = 3  # 检索出三张相似度最高的图片
imlist = []
for i ,index in enumerate(rank_score_n1[0:maxres]):
    imlist.append(rank_name_n1[i])
    # print(type(imgNames[index]))
    print("image names: " + str(rank_name_n1[i]) + " scores: %f" % rank_score_n1[i])
print("aHash--top %d images in order are: " % maxres, imlist)


imlist = []
for j ,index in enumerate(rank_score_n2[0:maxres]):
    imlist.append(rank_name_n2[j])
    # print(type(imgNames[index]))
    print("image names: " + str(rank_name_n2[j]) + " scores: %f" % rank_score_n2[j])
print("dHash--top %d images in order are: " % maxres, imlist)


imlist = []
for k ,index in enumerate(rank_score_n3[0:maxres]):
    imlist.append(rank_name_n3[k])
    # print(type(imgNames[index]))
    print("image names: " + str(rank_name_n3[k]) + " scores: %f" % rank_score_n3[k])
print("pHash--top %d images in order are: " % maxres, imlist)



