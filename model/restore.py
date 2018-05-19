import tensorflow as tf

import os
dir = os.path.dirname(os.path.realpath(__file__))

tf.reset_default_graph()

v1 = tf.get_variable("v1", [3])
v2 = tf.get_variable("v2", [5])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, dir + '/data-all')
    print("Model restored.")

    print("v1: {}".format(v1.eval()))
    print("v2: {}".format(v2.eval()))


