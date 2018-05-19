"""
This example show how to use Tensorflow eager exectuion to fit a simple
linear regression model using some synthesized data. Specifically, it 
illusrates how to define the forward path of the linear model and the loss
function, as well as how to obtain the gradients of the loss function with
respect to the variables and update the variables with the gradients.
"""

import argparse
import sys

import tensorflow as tf
import tensorflow.contrib.eager as tfe

class LinearModel(tfe.Network):
    """
    A TensorFlow linear regression model.

    Uses TensorFlow's eager execution.
    """
    def __init__(self):
        """Constructs a Linear Model object."""
        super(LinearModel, self).__init__()
        self._hidden_layer = self.track_layer(tf.layers.Dense(1))

    def call(self, xs):
        """Invoke the linear model.

        Args:
            xs: input features, as a tensor of size [batch_size, ndims].

        Returns:
            ys: the predictions of the linear model, as a tensor of size [batch_size]
        """
        return self._hidden_layer(xs)

   
def synthetic_dataset(w, b, noise_level, batch_size, num_batches):
    """tf.data.Dataset that yields synthetic data for linear regression."""
    def batch(_):
        x = tf.random_normal([batch_size, tf.shape(w)[0]])
        y = tf.matmul(x, w) + b + noise_level * tf.random_normal([])
        return x, y

    with tf.device("/device:CPU:0"):
        return tf.data.Dataset.range(num_batches).map(batch)

def fit(model, dataset, optimizer, verbose=False, logdir=None):
    """Fit the linear-regression model.
    
    Args:
        model: The Linear Model to fit.
        dataset: The tf.data.Dataset to use for training data
        optimizer: The Tensorflow Optimizer object to be used
        verbose: If true, will print out loss values at every iteration.
        logdir: The directory in which summaries will be written for Tensorboard.(Optional)
    """
    
    #The loss function to optimize.
    def mean_square_loss(xs, ys):
        return tf.reduce_mean(tf.square(model(xs) - ys))
   
    #Returns a function which differentiates f with respect to variables.
    loss_and_grads = tfe.implicit_value_and_gradients(mean_square_loss)

    tf.train.get_or_create_global_step()

    if logdir:
        summary_writer = tf.contrib.summay.create_file_writer(logdir)

    #Training loop.
    for i, (xs, ys) in enumerate(tfe.Iterator(dataset)):
        loss, grads = loss_and_grads(xs, ys)

        if verbose:
            print("Iteration {}: loss {}".format(i, loss.numpy()))
    
        optimizer.apply_gradients(grads, global_step = tf.train.get_global_step())

        if logdir:
            with summary_writer.as_default():
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("loss", loss)
    
def main(_):
    tfe.enable_eager_execution()
    
    #Ground-truth constants.
    true_w = [[-2.0], [4.0], [1.0]]
    true_b = [0.5]
    noise_level = 0.01

    #Training constants.
    batch_size = 64
    learning_rate = 0.1

    print("True w: {}".format(true_w))
    print("True b: {}".format(true_b))

    model = LinearModel()
    dataset = synthetic_dataset(true_w, true_b, noise_level, batch_size, 20)

    device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    print("Using device: {}".format(device))
    
    with tf.device(device):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        fit(model, dataset, optimizer, verbose = True, logdir = FLAGS.logdir)
    
    print("\n After training: w = {}".format(model.variables[0].numpy()))
    print("\n After training: b = {}".format(model.variables[1].numpy()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--logdir",
            type=str,
            default=None,
            help="logdir in which TensorBoard summaries will be written(optional).")
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


