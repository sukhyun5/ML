#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define Training
W1 = tf.Variable(tf.truncated_normal([784,512], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([512], stddev=0.1))

W2 = tf.Variable(tf.truncated_normal([512,10], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

h1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
y = tf.nn.softmax(tf.matmul(h1,W2) + b2)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))

train = tf.train.AdamOptimizer(0.001).minimize(loss)

# Make Session and Initialize Variables
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Run Training
for i in range(5000):
    batch = mnist.train.next_batch(128)
    sess.run(train, feed_dict = {x: batch[0], y_: batch[1]})

# Testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
