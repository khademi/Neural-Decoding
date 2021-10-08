from nilearn import image
import tensorflow as tf
import numpy as np

import nibabel as nib

data_folder = '/local-scratch/localhome/mkhademi/ds001499/derivatives/fmriprep'

filename = '/external1/gqa_dataset/code/sub-CSI1.tfrecords'
dataset = tf.data.TFRecordDataset(filenames = [filename]).shuffle(10000).batch(15)
writer = tf.io.TFRecordWriter(filename)

img4d = nib.load(data_folder + '/sub-CSI1/ses-01/func/sub-CSI1_ses-01_task-5000scenes_run-01_bold_space-T1w_preproc.nii.gz')
img4d_data = np.array(img4d.dataobj)
print(img4d_data.shape) # (71, 89, 72, 194)

for i in range(img4d_data.shape[3]):
    x = np.reshape(img4d_data[:,:,:,i], (-1))
    example = tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x)),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=x)), 

       }))
    writer.write(example.SerializeToString()) 
writer.close()
