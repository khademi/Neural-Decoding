from nilearn import image
import tensorflow as tf
from keras.applications.nasnet import preprocess_input
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import nibabel as nib
import json
import os

data_dir = '/local-scratch/localhome/mkhademi/BOLD5000_2.0/'
with open(data_dir + 'image_data/MSCOCO/annotations/' + 'instances_train2014.json') as json_data:
    coco_anns = json.load(json_data)
    json_data.close()

imagenet_anns = pd.read_csv(data_dir + 'image_data/LOC_train_solution.csv', sep = ',')
# print(imagenet_anns)
f = open(data_dir + 'image_data/LOC_synset_mapping.txt', 'r')
imagenet_categories = []
for x in f:
    imagenet_categories.append(x.split()[0])
f.close()

subjs = ['CSI1', 'CSI2', 'CSI3', 'CSI4']
seses = ['ses-01', 'ses-02', 'ses-03', 'ses-04', 'ses-05', 'ses-06', 'ses-07', 'ses-08',
         'ses-09', 'ses-10', 'ses-11', 'ses-12', 'ses-13', 'ses-14', 'ses-15']
         
sub = subjs[0]
imgnames = []
f = open(data_dir + sub + '_imgnames.txt', 'r')
for imgname in f:
    imgnames.append(imgname[:-1])
f.close()
img_dict = {}
coco_category_count = {}
imagenet_category_count = {}
f_coco = open(data_dir + 'image_data/coco_cat.txt', 'w')
f_imagenet = open(data_dir + 'image_data/imagenet_cat.txt', 'w')
for imgname in imgnames:
    if imgname[0]  == 'C':
        os.system('cp ' + data_dir + 'image_data/MSCOCO/images/train2014/' + imgname + ' ' + data_dir + 'image_data/drop_box/')
        f_coco.write(imgname +': ')
        img_id = int(imgname[15:27])
        img_dict[imgname] = np.zeros(90, dtype=np.int32)
        tmp_set = set()
        for i in range(len(coco_anns['annotations'])):
            if coco_anns['annotations'][i]['image_id'] == img_id:
                category_id = coco_anns['annotations'][i]['category_id'] - 1
                img_dict[imgname][category_id] = 1
                if not 'c_' + str(category_id) in tmp_set:
                    if 'c_' + str(category_id) in coco_category_count:
                        coco_category_count['c_' + str(category_id)] += 1
                    else:
                        coco_category_count['c_' + str(category_id)] = 1
                    tmp_set.add('c_' + str(category_id))
                f_coco.write('c_' + str(category_id) + ' ')
        f_coco.write('\n')
    if imgname[0]  == 'n' and (imgname[1] == '0' or imgname[1] == '1'):
        img_dict[imgname] = np.zeros(1000, dtype=np.int32)
        category_id = imagenet_categories.index(imgname[:9])
        img_dict[imgname][category_id] = 1
        f_imagenet.write(imgname +': ')
        if 'c_' + str(category_id) in imagenet_category_count:
            imagenet_category_count['c_' + str(category_id)] += 1
        else:
            imagenet_category_count['c_' + str(category_id)] = 1
        f_imagenet.write('c_' + str(category_id) + ' ')
        f_imagenet.write('\n')
#print(coco_category_count)
f_coco.close()
print(imagenet_category_count)
print(len(imagenet_category_count))
f_imagenet.close()

nasnet = NASNetLarge() 
filename = data_dir + 'image_data/bold5000_coco.tfrecords'
writer_coco = tf.io.TFRecordWriter(filename)

filename = data_dir + 'image_data/bold5000_imagenet.tfrecords'
writer_imagenet = tf.io.TFRecordWriter(filename)   
i = 0
for ses in seses:
    img4d = nib.load(data_dir + sub + '_GLMbetas-TYPEA-ASSUMEHRF_' + ses + '.nii.gz')
    img4d = np.array(img4d.dataobj)
    print(img4d.shape) # (71, 89, 72, 370) 
    img4d = np.nan_to_num(img4d, nan = 0.0)
    for j in range(img4d.shape[3]):
        x = np.reshape(img4d[:, :, :, j], (-1))
        imgname = imgnames[i]
        i += 1
        coco_imagenet = False
        coco_label = np.zeros(90, dtype=np.int32)
        imagenet_label = np.zeros(1000, dtype=np.int32) 
        if imgname[0] == 'C':
            img_path =  data_dir + 'image_data/MSCOCO/images/train2014/' + imgname 
            coco_label = img_dict[imgname]
            coco_imagenet = True
        if  imgname[0]  == 'n' and (imgname[1] == '0' or imgname[1] == '1'):
            img_path = data_dir + '/image_data/ILSVRC/Data/CLS-LOC/train/' + imgname[:9] + '/' + imgname
            imagenet_label = img_dict[imgname]
            coco_imagenet = True
        if  coco_imagenet:
            image = load_img(img_path, target_size=(331, 331))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)  
            yhat = nasnet.predict(image)
            example = tf.train.Example(features=tf.train.Features(feature={
                'x': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                'y': tf.train.Feature(float_list=tf.train.FloatList(value=yhat[0])),
                'y_coco': tf.train.Feature(int64_list=tf.train.Int64List(value=coco_label)),
                'y_imagenet': tf.train.Feature(int64_list=tf.train.Int64List(value=imagenet_label)) 
                }))
            if imgname[0] == 'C':
                writer_coco.write(example.SerializeToString())
            else:
                writer_imagenet.write(example.SerializeToString())
writer_coco.close()
writer_imagenet.close()
# -4218170.5
