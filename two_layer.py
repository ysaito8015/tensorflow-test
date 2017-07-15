import tensorflow as tf

# define weight
def weight(dtype = tf.float32, shape = []):
    initial = tf.trancated_normal(dtype = dtype, shape, stddev = 0.01)
    return tf.Variable(initial)

# define bias
def bias(dtype = tf.float32, shape = []):
    initial = tf.zeros(shape, dtype = dtype)
    return tf.Variable(initial)


# data points
Q = 5
P = 2
R = 1

# declare placeholder
X = tf.placeholder(dtype = tf.float32, shape = [None, Q])

# layer 1
W1 = weight(shape = [Q, P])
b1 = bias(shape = [P])
f1 = tf.matmul(X, W1) + b1

# layer 2
W2 = weight(shape = [P, R])
b2 = bias(shape = [R])
f2 = tf.matmul(f1, W2) + b2
