import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.random.normal(0, 0.05, (1,100)).astype(np.float32)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [1, 100])

l1 = add_layer(xs, 100, 50, activation_function=tf.nn.relu)
l2 = add_layer(l1, 50, 100, activation_function=tf.nn.relu)
prediction = add_layer(l2, 100, 2, activation_function=None)
# number of initialization
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    print(prediction_value)
