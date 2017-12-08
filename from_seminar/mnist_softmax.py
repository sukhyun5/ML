#!/usr/bin/python
# -*- coding: utf-8 -*-

# # MNIST Softmax

# In[ ]:

import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility


# In[ ]:

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset


# In[ ]:

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


# In[ ]:

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))


# In[ ]:

hypothesis = tf.matmul(X, W) + b

# define cost/loss & optimizer
#cost = -tf.reduce_mean(Y * tf.log(tf.nn.softmax(hypothesis)) + (1 - Y) * tf.log(1 - tf.nn.softmax(hypothesis)))

#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.nn.softmax(hypothesis)), reduction_indices=[1]))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))


#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:
"""
print ("train num_examples: ", mnist.train.num_examples)
print ("test num_examples: ", mnist.test.num_examples)

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')


# In[ ]:

print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))
"""

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    avg_cost = 0
    batch_xs, batch_ys = mnist.train.next_batch(100)
    c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
    avg_cost += c / 100
    if i % 100 == 0:
        print('Epoch:', '%04d' % (i + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))


"""
# In[ ]:

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
"""
