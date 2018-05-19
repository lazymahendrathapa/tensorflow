import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

"""
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28 * 28 pixel, 
we will then handle 28 sequences of 28 steps for every sample.
"""

#Training Parameters
learning_rate = 0.001
training_steps = 20000
batch_size = 128
display_step = 200

#Network parameters
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

#tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

#Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(X, weights, biases):

    # Unstack to get a list of timesteps tensors of shape (batch_size, num_input)
    X = tf.unstack(X, timesteps, 1)

    #Define lstm cells with tensorflow
    #Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    #Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X, dtype = tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

#Evalute model 
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_X, batch_Y = mnist.train.next_batch(batch_size)

        batch_X = batch_X.reshape((batch_size, timesteps, num_input))

        sess.run(train_op, feed_dict={X: batch_X, Y: batch_Y})

        if step % display_step == 0 or step == 1:
            #Calcuate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict = {X: batch_X, Y: batch_Y})

            print("Step: {}, Minibatch Loss: {:.4f}, Training Accuracy: {:.3f}".format(step, loss, acc))
    
    print("Optimization Finished!")

    #Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]

    print("Testing accuracy: {}".format(sess.run(accuracy, feed_dict={X: test_data, Y: test_label})))
