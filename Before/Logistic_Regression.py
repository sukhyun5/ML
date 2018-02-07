#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

# same params and variables initialization as log reg.
"""
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")

merged = tf.summary.merge_all()

# former inference is now used for combining inputs
def combine_inputs(X):
    return tf.matmul(X, W) + b


# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))
"""
x_data = [[1, 2],  # Dim: 6 X 2
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],     # Dim: 6 X 1
          [0],
          [0],
          [1],
          [1],
          [1]]
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=(tf.matmul(X, W) + b), labels=Y))
#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train_value = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
        


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(5, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    # convert categorical data
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [5, 1])

    return features, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(inference(X) > 0.5, tf.float32)

    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    print "init"
    #tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    #writer = tf.summary.FileWriter("./tensorflow", sess.graph)

    """
    print "before input"
    X, Y = inputs()

    print "before loss"
    total_loss = loss(X, Y)
    print "before train"
    train_op = train(total_loss)

    print "before Coordinator"
    #coord = tf.train.Coordinator()
    print "before queue"
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    """
    # actual training loop
    training_steps = 10001
    
    print "before for"
    for step in range(training_steps):
        #summary = sess.run([train_op])
        cost_val, _ = sess.run([cost, train_value], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            #print "loss: ", sess.run([total_loss])
            print (step, cost_val)

    print " before evaluate"
    #evaluate(sess, X, Y)

    #import time
    #time.sleep(5)

    #coord.request_stop()
    #coord.join(threads)
    #sess.close()
