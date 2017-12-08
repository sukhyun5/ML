#!/usr/bin/python
# -*- coding: utf-8 -*- 

import tensorflow as tf
import os

# 변수 그리고 model parameter들을 초기화 한다.
W = tf.Variable(tf.zeros([4, 3]), name="weights")
"""
array([ [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]], dtype=float32)

array(  [ 0.,  0.,  0.], dtype=float32)
"""
b = tf.Variable(tf.zeros([3]), name="bias")

# training loop 에서 사용되는 operation들을 정의한다.
def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    # X data를 inference model을 이용해 계산하고 그 결과를 return
    return tf.nn.softmax(combine_inputs(X))


def loss(X, Y):
    #return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded, batch_size=batch_size,
                                 capacity=batch_size * 50,
                                 min_after_dequeue=batch_size)

def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label =\
        read_csv(150, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])
    
    # convert categorical data
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]), 
        tf.equal(label, ["Iris-virginica"])
    ])), 0))
    
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))
    
    return features, label_number

  
def GD_train(total_loss):
    # 계산된 total loss 에 따라 model parameter들을 train / 조절 한다.
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    
def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))

#saver = tf.train.Saver()

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    
    tf.global_variables_initializer().run()
    
    X, Y = inputs()
    
    total_loss = loss(X, Y)
    
    train_op = GD_train(total_loss)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    training_steps = 1000
    
    # actual training Loop
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and Learning purposes, see how the Loss gets decremented thru training steps
        if step % 10 == 0:
            print ("loss: ", sess.run([total_loss]))
            #print (sess.run(inference(X)), sess.run([total_loss]))
            #print (sess.run(W), sess.run(b), sess.run(inference(X)), sess.run([total_loss]))
            
    evaluate(sess, X, Y)
    
    writer =tf.summary.FileWriter("./name_scope_1", sess.graph)
    
    import time
    time.sleep(5)
       
      
    coord.request_stop()
    coord.join(threads)
    sess.close()

