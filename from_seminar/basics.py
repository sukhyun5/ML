#!/usr/bin/python
# -*- coding: utf-8 -*-

# # Getting Started With TensorFlow
# https://www.tensorflow.org/get_started/get_started

# ## Check TF version

# In[ ]:

import tensorflow as tf
tf.__version__


# ## Hello TensorFlow!

# In[ ]:

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, TensorFlow!")

# start a TF session
sess = tf.Session()

sess.run(tf.global_variables_initializer())
# run the op and get result
print(sess.run(hello))


# ## Computational Graph

# In[ ]:

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)


# In[ ]:

print("node1:", node1, "node2:", node2)
print("node3: ", node3)


# ![Graph](https://www.tensorflow.org/images/getting_started_adder.png)

# In[ ]:

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))


# ## Placeholder

# In[ ]:

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))


# In[ ]:

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b:4.5}))


# ## Tensors

# In[ ]:

# t = 3
t = tf.constant(3) # a rank 0 tensor; this is a scalar with shape []
print(sess.run(t))


# In[ ]:

# t = [1., 2., 3.]
t = tf.constant([1.,2.,3.]) # a rank 1 tensor; this is a vector with shape [3]
print(sess.run(t))


# In[ ]:

# t = [[1., 2., 3.], [4., 5., 6.]]
t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # a rank 2 tensor; a matrix with shape [2, 3]
print(sess.run(t))


# In[ ]:

# t = [[[1., 2., 3.]], [[7., 8., 9.]]]
t = tf.constant([[[1., 2., 3.]], [[7., 8., 9.]]]) # a rank 3 tensor with shape [2, 1, 3]
print(sess.run(t))


# In[ ]:



