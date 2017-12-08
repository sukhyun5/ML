#!/usr/bin/python
# -*- coding: utf-8 -*- 

import tensorflow as tf


# 1) Graph build
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train*W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 2) Execute the Graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3) update the variables
for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))

"""
# 변수 그리고 model parameter들을 초기화 한다.

# training loop 에서 사용되는 operation들을 정의한다.
def inference(X):
    # X data를 inference model을 이용해 계산하고 그 결과를 return
    pass
    
def loss(X, Y):
    # training data X와 실제값 Y의 차 인 loss를 계산한다.
    pass
    
def inputs():
    # trainint data X를 입력받고 기대값인 Y를 출력해낸다.
    pass
    
def train(total_loss):
    # 계산된 total loss 에 따라 model parameter들을 train / 조절 한다.
    pass
    
def evaluate(sess, X, Y):
    # evaluate the resulting trained model
    # trained model을 이용해 결과를 계산한다.
    pass
    
    
# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    
    # tf.initialize_all_variables().run() <-- deprecated
    tf.global_variables_initializer().run()
    
    # i) input training data
    # ii) Execute inference model on training data
    # iii) Compute loss
    # iv) Adjust model parameters
    
    # i) input training data
    X, Y = inputs()
    
    # ii) Compute loss
    total_loss = loss(X, Y)
    
    # iii) Adjust model parameters
    train_op = train(total_loss)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # actual training Loop
    training_steps = 1000
    
    initial_step = 0
    
    # verify if we don't have a checkpoint saved already
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
        
    # actual training Loop
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and Learning purposes, see how the Loss gets decremented thru training steps
        if step % 10 == 0:
            print "loss: ", sess.run([total_loss])
            
            
    evaluate(sess, X, Y)
    
    coord.request_stop()
    coord.join(threads)
    sess.close()
"""
