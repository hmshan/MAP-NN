
import tensorflow as tf
import numpy as np
from models import *
import h5py
import pydicom as dicom
import time
import os
import logging
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0'

region = 'ABD'  # or 'CH'

image_name = 'test_images/L506_QD_1_1.CT.0004.0087.2015.12.22.20.46.00.71702.358798544.IMA' # LDCT image

#image_name = 'test_images/L506_FD_1_1.CT.0002.0087.2015.12.22.20.19.52.894480.358591878.IMA' # NDCT image


input_width = 512
input_height = 512

Num_CLONE = 5

# generator networks
X = tf.placeholder(dtype=tf.float32, shape=[1, input_width, input_height, 1])

with tf.variable_scope('generator_model', reuse=tf.AUTO_REUSE) as scope:
    Y = MAP_NN_all(X, padding='valid', D = Num_CLONE) # Y stores all intermediate denoised images and input itself.





sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, './pretrained_models/Trained_%s_Model.ckpt'%str(region))


img_slice = dicom.read_file(image_name)
img_pixel = img_slice.pixel_array
if region == 'ABD':
	img_pixel = (img_pixel - 1024.0 + 160.0) / 400.0 
else:
	img_pixel = (img_pixel - 1024.0 + 1350.0) / 1500.0
	
img_pixel[img_pixel>1.0] = 1.0
img_pixel[img_pixel<0.0] = 0.0


inputdata = img_pixel
inputdata = np.expand_dims(inputdata, axis=3)
inputdata = np.expand_dims(inputdata, axis=0)
outputdata = sess.run(Y, feed_dict = {X : inputdata})

plt_data = []
for img in outputdata:
    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    img = np.squeeze(img)
    plt_data.append(img)


fig_titles = ['LDCT', 'D = 1', 'D = 2', 'D = 3', 'D = 4', 'D = 5']
plt.figure()
f, axs = plt.subplots(figsize=(40, 40), nrows=1, ncols=6)

for i, img in enumerate(plt_data):
    axs[i].imshow(img, cmap=plt.cm.gray)
    axs[i].set_title(fig_titles[i], fontsize=30)
    axs[i].axis('off')
plt.savefig('test_result.pdf', bbox_inches = 'tight', pad_inches=0.1)
plt.show()

sess.close()


