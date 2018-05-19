import tensorflow as tf

sess = tf.InteractiveSession()

Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant([ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ])

print("Scalar: {} ".format(Scalar.eval()))
print("Vector: {} ".format(Vector.eval()))
print("Matrix: {} ".format(Matrix.eval()))
print("Tensor: {} ".format(Tensor.eval()))

Matrix_one = tf.constant([[1,2,3], [2,3,4], [3,4,5]])
Matrix_two = tf.constant([[2,2,2], [2,2,2], [2,2,2]])

# two ways of addition
first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two

print("Defined using tensorflow function: {}".format(first_operation.eval()))
print("Defined using normal expression: {}:".format(second_operation.eval()))

# tf.matmul
Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

first_operation = tf.matmul(Matrix_one, Matrix_two)
print("Defined using tensorflow function: {}".format(first_operation.eval()))

# variable
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

state.initializer.run()
print("State value {} ".format(state.eval()))

for i in range(3):
    update.eval()
    print("State value {} for i {}".format(state.eval(), i))

#placeholder
a = tf.placeholder(tf.float32)
b = a * 2

print("Value of b: {}".format(b.eval(feed_dict={a:[3.5, 1.0]})))
print("Value of b: {}".format(b.eval(feed_dict={a:3.5})))

sess.close()
