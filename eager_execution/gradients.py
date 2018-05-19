import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

def square(x):
    return tf.multiply(x, x)

grad = tfe.gradients_function(square) #automatic differentiation

print(square(3.))
print(grad(3.))

gradgrad = tfe.gradients_function(lambda x : grad(x)[0])
print(gradgrad(3.))

def abs(x):
    return x if x > 0. else -x

grad = tfe.gradients_function(abs)
print(grad(2.0))
print(grad(-2.0))

#Custom gradients for an operation, or for a function.

def log1pexp(x):
    return tf.log(1 + tf.exp(x))

grad_log1pexp = tfe.gradients_function(log1pexp)

#The gradient computation works fine at x = 0.
print(grad_log1pexp(0.))
#[0.5]
# However it returns a `nan` at x = 100 due to numerical instability.
print(grad_log1pexp(100.))
# [nan]

"""
We can use a custom gradient fro the above function that analytically
simplifies the gradient expression. Gradient function implementation below
reuses an expression(tf.exp(x)) that was computed during the forward pass, 
making the gradient computation more efficient by avoiding redundant 
computation.
"""

@tfe.custom_gradient
def log2pexp(x):
    e = tf.exp(x)
    def grad(dy):
        return dy * (1 - 1 / (1 + e))
    return tf.log(1 + e), grad

grad_log2pexp = tfe.gradients_function(log2pexp)

#Gradient at x = 0 works as before.
print(grad_log2pexp(0.))

#[0.5]
# And now gradient computation at x = 100 works as well.
print(grad_log2pexp(100.))
#[1.0]

