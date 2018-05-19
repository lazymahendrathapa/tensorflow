import tensorflow as tf

sess = tf.InteractiveSession()

"""
Internally, Tensorflow represents tensors as n-dimensional arrays of base datatypes.
"""
a = tf.constant([2]) #Tensor
b = tf.constant([3]) 
c = tf.add(a,b) #Node

print(a.eval())
print(c.eval())

sess.close()
