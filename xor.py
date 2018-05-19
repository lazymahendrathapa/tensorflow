import tensorflow as tf

x = tf.placeholder(tf.float32, shape = [4,2], name = 'x-input')
y = tf.placeholder(tf.float32, shape = [4,1], name = 'y-input')

theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "theta1")
theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "theta2")

bias1 = tf.Variable(tf.zeros([2]), name = "bias1")
bias2 = tf.Variable(tf.zeros([1]), name = "bias2")

with tf.name_scope("layer2") as scope:
    A2 = tf.sigmoid(tf.matmul(x, theta1) + bias1)

with tf.name_scope("layer3") as scope:
    Hypothesis = tf.sigmoid(tf.matmul(A2, theta2) + bias2)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(((y * tf.log(Hypothesis)) + ((1-y) * tf.log(1.0 - Hypothesis))) * -1 )

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(100000):
        sess.run(train_step, feed_dict = {x:XOR_X, y:XOR_Y})

        if i % 10000 == 0:
            print("Epoch ", i)
            print("Hypothesis ", sess.run(Hypothesis, feed_dict={x: XOR_X, y:XOR_Y}))
            print("Theta1 ", sess.run(theta1))
            print("Bias1 ", sess.run(bias1))
            print("Theta2 ", sess.run(theta2))
            print("Bias2 ", sess.run(bias2))
            print("Cost ", sess.run(cost, feed_dict={x: XOR_X, y:XOR_Y}))

