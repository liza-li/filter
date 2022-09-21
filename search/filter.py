import jsonlines
import json



index = '/datasets/coco/annotations/person_keypoints_train2017.json'
img_id = '/home/zeli/projects/search/img_id.txt'
filter_json = '/home/zeli/projects/search/filter_json.json'
 
s = 128  #bbox size
area_thr = s*s/4*3


json_list = []
with open(index, 'r') as f:
    for item in jsonlines.Reader(f):
            json_list.append(item)
            
img_ids = [] 
clean_ids = []           
for i in json_list:
    count=0
    for j in i['annotations']:
        if j['category_id']!=1:
            continue
        if j['iscrowd']!=0:
            continue
        area = j['bbox'][2] * j['bbox'][3] #w*h
        if area < area_thr:
            continue
        
        print("已读取",count+1,"张img")
        count=count+1
        img_ids.append(j['image_id'])
         
        with open(filter_json, 'a') as write_f:
	        json.dump(j, write_f, indent=4, ensure_ascii=False)
           
         
        
        
       


'''
img_ids.sort()        
print("end,共",count,"张合格img") 

for i in range(1,len(img_ids),2):   #循环数组，验证前后是否相同，由于原始出现两次，因此可跳跃判断
            if img_ids[i-1] != img_ids[i]  :
                clean_ids.append(img_ids[i])
            if (i+2) == len(img_ids):   #判断单一元素在排序后数组的最后面
                clean_ids.append(img_ids[i])


'''


    

print("正在写入...")
for temp in img_ids:
    txt_file = open(img_id, "a", encoding="utf-8")  # 以写的格式打开先打开文件
    
    txt_file.write(str(temp))
    txt_file.write("\n")
    txt_file.close()


print("over")

  

                    

        