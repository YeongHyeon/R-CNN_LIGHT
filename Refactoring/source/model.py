import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

def convolution_neural_network(x, y_, training=None, height=None, width=None, classes=None):

    print("\n** Initialize CNN Layers")

    channel = 3
    x_data = tf.reshape(x, [-1, height, width, channel])
    print("Input: "+str(x_data.shape))

    #CONV

    flatten_layer = flatten(inputs=drop_3)

    full_con = fully_connected(inputs=flatten_layer, num_outputs=classes, activate_fn=None)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=full_con, labels=y_)
    mean_loss = tf.reduce_mean(cross_entropy) # Equivalent to np.mean

    train_step = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(mean_loss)

    prediction = tf.contrib.layers.softmax(full_con) # Want to prediction Use this!
    correct_pred = tf.equal(tf.argmax(full_con, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return train_step, accuracy, mean_loss, prediction
