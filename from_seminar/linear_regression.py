#!/usr/bin/python
# -*- coding: utf-8 -*-

# # Linear Regression

# In[ ]:

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility


# In[ ]:

# X and Y data
x_train = [[1.0, 2.0, 3.0]]
y_train = [[1, 2, 3]]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
W = tf.Variable(tf.random_normal([1,1]), name='weight')
b = tf.Variable(tf.random_normal([1,1]), name='bias')


# #### Do It Yourself!

# In[ ]:

# Our hypothesis XW+b
hypothesis = x_train * W + b


# In[ ]:

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# In[ ]:

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# Learns best fit W:[ 1.],  b:[ 0.]


# In[ ]:



