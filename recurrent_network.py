import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28 * 28 px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training parameters
learning_rate = 0.001
training_steps = 20000
batch_size = 128
display_step = 200

num_input = 28
timesteps = 28
num_hidden =  128
num_classes = 10

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'out' : tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases = {
    'out' : tf.Variable(tf.random_normal([num_classes]))
}

def RNN(X, weights, biases):

    X = tf.unstack(X, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)

    outputs, states = rnn.static_rnn(lstm_cell, X,  dtype = tf.float32) # tuple of outputs, and states from hidden layer.

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variable(i.e assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_X, batch_Y = mnist.train.next_batch(batch_size)
        #Reshape data to get 28 seq of 28 elements
        batch_X = batch_X.reshape((batch_size, timesteps, num_input))
        # Run optimization op
        sess.run(train_op, feed_dict = {X: batch_X, Y: batch_Y})

        if step % display_step == 0 or step == 1:
            #Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict = {X: batch_X, Y : batch_Y})
            print("Step: {}, Minibatch Loss: {:.4f}, {:.3f}".format(step, loss, acc))
    
    print("Optimization Finished!")

    #Calcuate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]

    print("Testing Accuracy: {}".format(sess.run(accuracy, feed_dict = {X: test_data, Y: test_label})))
