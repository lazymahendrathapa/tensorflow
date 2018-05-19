"""
Eager execution is an imperative, define-by-run interface where operations
are executed immediately as they are called from Python.

The benefits of eager execution include:
    1) Fast debugging with immediate run-time errors and integration with 
       Python tools.
    2) Support for dynamic models using easy-to-use Python control flow.
    3) Strong support for custom and higher-order gradients.
    4) Almost all of the available TensorFlow operations.
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

x = [[2.]]
m = tf.matmul(x,x)

"""
When you enable eager exection, operations execute immediately and return 
their values to Python without requiring a Session.run(). It's straightforward
to inspect intermediate results with print or the Python debugger.
"""
print(m)

a = tf.constant(12)
counter = 0

while not tf.equal(a,1):
    if tf.equal(a % 2, 0):
        a = a / 2
    else:
        a = 3 * a + 1
    print(a)
