import os 

import tensorflow as tf
from tf_flags.main import main as m

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 60, "Batch Size[60]")

def main(_):
    config = flags.FLAGS
    m(config)

if __name__ == "__main__":
    tf.app.run()
