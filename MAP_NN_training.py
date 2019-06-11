#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from models import *
import h5py
import time
import os
import logging
from sklearn.utils import shuffle


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


input_width = 64
input_height = 64

output_width = 64
output_height = 64

batch_size = 128

learning_rate = 1e-4

beta1 = 0.9
beta2 = 0.999

Num_CLONE = 5
lambda_p = 50
edge_p = 50

region = 'ABD'  # or 'CH'

num_epoch = 80
disc_iters = 4

Methodname = 'Mayo_' + str(region) + '_model_R' + str(Num_CLONE) + '_p' + str(lambda_p) + '_edge_p' + str(edge_p)

Networkfolder = Methodname

if not os.path.exists('Mayo_logs'):
	os.mkdir('Mayo_logs')

if not os.path.exists('Networks'):
	os.mkdir('Networks')

logfilename = 'Mayo_logs/' + Networkfolder + '.log'

###################################################

if os.path.exists('Networks/' + Networkfolder):
	print('Warning!  Network/' + Networkfolder + ' exists!!!!!')
	raise NameError('Folder exists')

if os.path.exists(logfilename):
	print('Warning! Log File has existed!!!!')
	raise NameError('Log FIle exists!!!')

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(message)s',
                    filename=logfilename,
                    filemode='w')

logging.info('Epoch, batch, time, disc_loss, difference, mse_cost, gen_loss, gen_cost, is_training')

# generator networks
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_width, input_height, 1])

with tf.variable_scope('generator_model', reuse=tf.AUTO_REUSE) as scope:
    Y_ = MAP_NN(X, padding='valid', D = Num_CLONE)

real_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_width, output_height, 1])

alpha = tf.random_uniform(shape=[batch_size,1],
                            minval=0.,
                            maxval=1.)

# discriminator networks
with tf.variable_scope('discriminator_model') as scope:
    disc_real = discriminator_model(real_data)
    scope.reuse_variables()
    disc_fake = discriminator_model(Y_)
    interpolates = alpha*tf.reshape(real_data, [batch_size, -1]) + (1-alpha)*tf.reshape(Y_, [batch_size, -1])
    interpolates = tf.reshape(interpolates, [batch_size, output_width, output_height, 1])
    gradients = tf.gradients(discriminator_model(interpolates), [interpolates])[0]

# generator loss
gen_cost = -tf.reduce_mean(disc_fake)

# discriminator loss
difference = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# gradient penalty
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)

# Final disc objective function
disc_loss = difference + 10 * gradient_penalty   # add gradient constraint to discriminator loss

gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_model')
disc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_model')

mse_cost = tf.reduce_mean(tf.squared_difference(Y_, real_data))

edge_cost = tf.reduce_mean(tf.squared_difference(tf.image.sobel_edges(Y_), tf.image.sobel_edges(real_data)))

gen_loss = gen_cost +   lambda_p * mse_cost + edge_p * edge_cost

# optimizer
lr = tf.placeholder(tf.float32, shape=[])
gen_train_op = tf.train.AdamOptimizer(learning_rate=lr,
                                        beta1=beta1,
                                        beta2=beta2).minimize(gen_loss, var_list=gen_params)

disc_train_op = tf.train.AdamOptimizer(learning_rate=lr,
                                        beta1=beta1,
                                        beta2=beta2).minimize(disc_loss, var_list=disc_params)



# training
sess = tf.Session()

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver(max_to_keep = 3)

def normalization(x, region = 'ABD'):
    if region == 'ABD':
    	lower = -160.0
    	upper = 240.0
    else:
    	lower = -1350.0
    	upper = 150.0

    x = (x - 1024.0 - lower) / (upper - lower)
    x[x<0.0] = 0.0
    x[x>1.0] = 1.0
    return np.transpose(x, (0, 2, 3, 1))


print('Loading dataset')
print('Warning: This is a lite dataset for testing code. The model should be sub-optimal!')
f = h5py.File('data/Mayo_lite_training.h5', 'r')
data, label = np.array(f['data']), np.array(f['label'])
f.close()

f = h5py.File('data/Mayo_lite_testing.h5', 'r')
test_data, test_label = np.array(f['data']), np.array(f['label'])
f.close()

print("Start training ... ")
for iteration in xrange(num_epoch):

    val_lr = learning_rate / np.sqrt(iteration + 1)
    data, label = shuffle(data, label)
    num_batches = data.shape[0] // batch_size

    for i in xrange(num_batches):
        start_time = time.time()
        # discriminator
        for j in xrange(disc_iters):
            idx = np.random.permutation(data.shape[0])
            batch_data = data[idx[:batch_size]]
            batch_label = label[idx[:batch_size]]
            sess.run([disc_train_op], feed_dict={real_data: normalization(batch_label, region),
                                                         X: normalization(batch_data, region),
                                                        lr: val_lr})

        batch_data = data[i*batch_size : (i+1)*batch_size]
        batch_label = label[i*batch_size : (i+1)*batch_size]

        # generator
        _disc_loss, _difference, _mse_cost, _gen_loss, _gen_cost, _ = sess.run([disc_loss, difference,  mse_cost,
                                                         gen_loss, gen_cost, gen_train_op],
                                                        feed_dict={real_data: normalization(batch_label, region),
                                                                           X: normalization(batch_data, region),
                                                                          lr: val_lr})

        t = time.time() - start_time
        print('Epoch: %d - %d - disc_loss: %.6f - gen_loss: %.6f - mse_loss: %.6f'%(
         iteration, i, _disc_loss , _gen_loss  , _mse_cost ))

        logging.info('%d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, 1'
            %(iteration, i, t, _disc_loss, _difference,  _mse_cost ,  _gen_loss, _gen_cost))



    _mse_cost = 0.0
    _disc_loss = 0.0
    _difference = 0.0
    _gen_loss = 0.0
    _gen_cost = 0.0

    start_time = time.time()
    test_num_batch = test_data.shape[0] // batch_size
    for j in xrange(test_num_batch):
        batch_data = test_data[j*batch_size:(j+1)*batch_size]
        batch_label = test_label[j*batch_size:(j+1)*batch_size]
        t_disc_loss, t_difference,  t_mse_cost,  t_gen_loss, t_gen_cost = sess.run([disc_loss, difference,  mse_cost,
                                                 gen_loss, gen_cost],
                                                feed_dict={real_data: normalization(batch_label, region),
                                                                   X: normalization(batch_data, region)})
        _mse_cost += t_mse_cost
        _disc_loss += t_disc_loss
        _difference += t_difference
        _gen_loss += t_gen_loss
        _gen_cost += t_gen_cost

    t = time.time() - start_time
    _mse_cost /= (test_num_batch)
    _disc_loss /= (test_num_batch)
    _difference /= (test_num_batch)
    _gen_loss /= (test_num_batch)
    _gen_cost /= (test_num_batch)
    print('Test....Epoch: %d - %d - disc_loss: %.6f - gen_loss: %.6f  - mse_loss: %.6f'%(
    iteration, i, _disc_loss, _gen_loss, _mse_cost ))

    logging.info('%d, %d, %.6f, %.6f,  %.6f,  %.6f,  %.6f,  %.6f, 0'
    %(iteration, i, t, _disc_loss, _difference,  _mse_cost,  _gen_loss, _gen_cost))

    if not os.path.exists('Networks/' + Networkfolder):
        os.mkdir('Networks/' + Networkfolder)
    saver.save(sess, 'Networks/'+ Networkfolder +'/MAP_NN_' + repr(iteration) + '.ckpt')

sess.close()

