
import tensorflow as tf

def MAP_NN(inputs, padding = 'valid', D = 5):
    '''
    MAP-NN 

    inputs: N x input_width x input_height x 1
    '''
    for _ in xrange(D):
 
        #inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_width, input_height, 1])
        outputs1 = tf.layers.conv2d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False) 
        outputs2 = tf.nn.relu(outputs1)
        
        outputs2 = tf.layers.conv2d(outputs2, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
        outputs3 = tf.nn.relu(outputs2) 

        outputs3 = tf.layers.conv2d(outputs3, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
        outputs4 = tf.nn.relu(outputs3)
        
        outputs4 = tf.layers.conv2d(outputs4, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
        outputs5 = tf.nn.relu(outputs4)
        
        outputs5 = tf.layers.conv2d_transpose(outputs5, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv5', use_bias=False)
        outputs5 = tf.concat([outputs3, outputs5], 3)
        outputs5 = tf.nn.relu(outputs5)

        outputs5 = tf.layers.conv2d(outputs5, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose1', use_bias=False)
        outputs6 = tf.nn.relu(outputs5) 

        outputs6 = tf.layers.conv2d_transpose(outputs6, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv6', use_bias=False)
        outputs6 = tf.concat([outputs2, outputs6], 3)
        outputs6 = tf.nn.relu(outputs6)

        outputs6 = tf.layers.conv2d(outputs6, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose2', use_bias=False)
        outputs7 = tf.nn.relu(outputs6) 

        outputs7 = tf.layers.conv2d_transpose(outputs7, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv7', use_bias=False)
        outputs7 = tf.concat([outputs1, outputs7], 3)
        outputs7 = tf.nn.relu(outputs7)

        outputs7= tf.layers.conv2d(outputs7, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose3', use_bias=False)
        outputs8 = tf.nn.relu(outputs7) 

        outputs8 = tf.layers.conv2d_transpose(outputs8, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv8', use_bias=False)
        inputs = tf.nn.relu(inputs + outputs8)

        inputs = tf.clip_by_value(inputs, 0.0, 1.0)
    
    return inputs 


def MAP_NN_all(inputs, padding='valid', D = 5):
    '''
    MAP-NN for testing. It returns all intermediates results
    '''
    res = []
    res.append(inputs)
    for _ in xrange(D):
 
        outputs1 = tf.layers.conv2d(inputs, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)
        outputs2 = tf.nn.relu(outputs1)
        
        outputs2 = tf.layers.conv2d(outputs2, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)
        outputs3 = tf.nn.relu(outputs2) 

        outputs3 = tf.layers.conv2d(outputs3, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)
        outputs4 = tf.nn.relu(outputs3)
        
        outputs4 = tf.layers.conv2d(outputs4, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)
        outputs5 = tf.nn.relu(outputs4)
        
        outputs5 = tf.layers.conv2d_transpose(outputs5, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv5', use_bias=False)
        outputs5 = tf.concat([outputs3, outputs5], 3)
        outputs5 = tf.nn.relu(outputs5)

        outputs5 = tf.layers.conv2d(outputs5, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose1', use_bias=False)
        outputs6 = tf.nn.relu(outputs5) 

        outputs6 = tf.layers.conv2d_transpose(outputs6, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv6', use_bias=False)
        outputs6 = tf.concat([outputs2, outputs6], 3)
        outputs6 = tf.nn.relu(outputs6)

        outputs6 = tf.layers.conv2d(outputs6, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose2', use_bias=False)
        outputs7 = tf.nn.relu(outputs6) 

        outputs7 = tf.layers.conv2d_transpose(outputs7, 32, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv7', use_bias=False)
        outputs7 = tf.concat([outputs1, outputs7], 3)
        outputs7 = tf.nn.relu(outputs7)

        outputs7= tf.layers.conv2d(outputs7, 32, 1, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='transpose3', use_bias=False)
        outputs8 = tf.nn.relu(outputs7) 

        outputs8 = tf.layers.conv2d_transpose(outputs8, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv8', use_bias=False)
        inputs = tf.nn.relu(inputs + outputs8)

        inputs = tf.clip_by_value(inputs, 0.0, 1.0)

        res.append(inputs)
    return res 


def leaky_relu(inputs, alpha):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)     


def discriminator_model(inputs):

    outputs = tf.layers.conv2d(inputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', strides=(2,2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.dense(outputs, units=1024, name='dense1')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.dense(outputs, units=1, name='dense2')

    return outputs