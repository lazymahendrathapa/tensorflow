import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values # Convert a categorical varaible into dummy variables.
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

"""
Placeholders -> This features of tensorflow allows us to create an algorithm which accepts
data and know something about the shape of data without knowing the amount of data going in.
"""
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                        mean = 0,
                                        stddev = 0.01,
                                        name = "weights"))

bias = tf.Variable(tf.random_normal([1, numLabels], 
                                    mean = 0,
                                    stddev = 0.01,
                                    name = "bias"))

apply_weights_OP = tf.matmul(X, weights, name = "apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name = "add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name = "activation")

# Training
# Number of Epochs in our training
numEpochs = 700

#Defining our learning rate iterations (decay)
learningRate = tf.train.exponential_decay(learning_rate = 0.0008,
                                            global_step = 1,
                                            decay_steps = trainX.shape[0],
                                            decay_rate = 0.95,
                                            staircase=True)

#Defining our cost function - Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")

#Defining our Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

#Create a tensorflow session
sess = tf.Session()

#Initialize our weights and biases variables.
init_OP = tf.global_variables_initializer()

#Initialize all tensorflow variables
sess.run(init_OP)

correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

#Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)

#Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

#Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

#Summary ops to check how variables(W,b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary Writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)
# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

#Training epochs
for i in range(numEpochs):
    if i > 1 and diff < 0.0001:
        print("change in cost {}; convergence".format(diff))
    else:
        step = sess.run(training_OP, feed_dict = {X: trainX, yGold: trainY})
        
        if i % 10 == 0:
            epoch_values.append(i)
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict = {X:trainX, yGold: trainY})
            accuracy_values.append(train_accuracy)
            cost_values.append(newCost)

            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step {}, training accuracy {}, cost {}, change in cost {}".format(i, train_accuracy, newCost, diff))

#How well do we perform on held-out test data?
print("Final accuracy on test set: {}".format(str(sess.run(accuracy_OP, feed_dict = {X:testX, yGold: testY}))))

plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()


