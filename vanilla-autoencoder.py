"""
Build a 2 layers auto-encoder with Tensorflow to compress images to a lower 
latent space and then reconstruct them.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 20000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784

X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1' : tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2' : tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1' : tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2' : tf.Variable(tf.random_normal([num_hidden_1, num_input]))
}

biases = {
    'encoder_b1' : tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2' : tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1' : tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2' : tf.Variable(tf.random_normal([num_input]))
}

# Building the encoder
def encoder(X):
    # Encoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    return layer_2

def decoder(X):
    # Decoder Hidden layer with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Lables) are the input data.
y_true = X

#Define loss and optimizer, minimize the squared error 
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for i in range(1, num_steps + 1):

        batch_X,_ = mnist.train.next_batch(batch_size)

        _, l = sess.run([optimizer, loss], feed_dict={X:batch_X})

        if i % display_step == 0 or i == 1:
            print("Step: {}, Minibatch Loss: {}".format(i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))

    for i in range(n):
        batch_X, _ = mnist.test.next_batch(n)

        g = sess.run(decoder_op, feed_dict = {X: batch_X})

        for j in range(n):
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j+1) * 28] = batch_X[j].reshape([28,28])

        for j in range(n):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j+1) * 28] = g[j].reshape([28,28])

    print("Original Images")
    plt.figure(figsize= (n,n))
    plt.imshow(canvas_orig, origin="upper", cmap = "gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize = (n,n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

