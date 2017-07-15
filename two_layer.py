import tensorflow as tf
import numpy as np

# define weight
def weight(shape = []):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

# define bias
def bias(dtype = tf.float32, shape = []):
    initial = tf.zeros(shape, dtype = dtype)
    return tf.Variable(initial)

# define sigmoid function
def sigmoid(x):
    return (1/(1 + tf.exp(-x)))

# data points
Q = 5
P = 2
R = 1

# setting interactiveSession()
# tf.InteractiveSession() ( The only difference with a regular Session is that an InteractiveSession installs itself as the default session on construction. )
# https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
sess = tf.InteractiveSession()

# declare placeholder
X = tf.placeholder(dtype = tf.float32, shape = [None, Q])

# layer 1
# tf.matmul ( dot product )
# https://www.tensorflow.org/api_docs/python/tf/matmul
W1 = weight(shape = [Q, P])
b1 = bias(shape = [P])
f1 = tf.matmul(X, W1) + b1
sigm = sigmoid(f1)

# layer 2
W2 = weight(shape = [P, R])
b2 = bias(shape = [R])
f2 = tf.matmul(sigm, W2) + b2

# If there is tf.Variables in the code, you should initialize the map.
init_op = tf.global_variables_initializer()
sess.run(init_op)

# output
# tf.reshape reshape( tensor, shape, name=None)
# https://www.tensorflow.org/api_docs/python/tf/reshape
# np.reshate ( resape(a, newshape, order='C')
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
y = sess.run(f2, {X: np.array([1,2,2,5,2]).astype(np.float32).reshape(1,5) })

# sess.run returns np.array

# hidden layer's output
#h = sess.run(sigm, {X: data })

