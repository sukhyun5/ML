#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

tf.set_random_seed(777)


# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
# tf.multiply(x_train, W) = tf.matmul(W, x_train) = x_train * b

# const/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launce the graph in a session
sess = tf.Session()
# Initialize global variables in the graph.
sess.run(tf.global_variables_initializer())

"""
# Fit the line
for step in range(2001):
    #sess.run(train)
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
            feed_dict={X: [1,2,3], Y: [1,2,3]})

    if step % 20 == 0:
        #print(step, sess.run(cost), sess.run(W), sess.run(b))
        print(step, cost_val, W_val, b_val)

print (sess.run(hypothesis, feed_dict={X: [5]}))
print (sess.run(hypothesis, feed_dict={X: [2.5]}))
print (sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
"""

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], 
            feed_dict={X: [1,2,3,4,5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print (step, cost_val, W_val, b_val)

print (sess.run(hypothesis, feed_dict={X: [5]}))
print (sess.run(hypothesis, feed_dict={X: [2.5]}))
print (sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
