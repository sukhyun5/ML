#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

# 1) Define graph
print "1"
graph = tf.Graph()

print "2"
# 2) Use defined 'graph' 
with graph.as_default():
    print "in the graph"
    
    # 2-1) define variables 
    with tf.name_scope("variables"):
        print "in the variables"
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

    # 2-2) define transformation
    with tf.name_scope("transformation"):
        print "in the transformation"
        #) define "input"
        with tf.name_scope("input"):
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
        #) define "middle layer"
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")
        #) define "output"
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")

    # 2-3) define update
    with tf.name_scope("update"):
        print "in the update"
        update_total = total_output.assign_add(output)
        increment_step = global_step.assign_add(1)

    # 2-4) define summary
    with tf.name_scope("summaries"):
        print "in the sumaries"
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")

        print "==> ", tf.summary.scalar("output_summary", output)
        print "==> ", tf.summary.scalar("total_summary", update_total)
        print "==> ", tf.summary.scalar("average_summary", avg)

    # 2-5) initialize variables
    with tf.name_scope("global_ops"):
        print "in the global_ops"
        init = tf.global_variables_initializer()
        merged_summaries = tf.summary.merge_all()

print "before session"
sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./name_scope_1', graph)

print "before run"
sess.run(init)

def run_graph(input_tensor):
    print "in the run_graph"
    feed_dict = {a: input_tensor}
    #_, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    sess.run(feed_dict=feed_dict)
    print "step : %d" ,step
    writer.add_summary(summary, global_step=step)


print "before run_graph"
run_graph([2, 8])
print "after run_graph"
"""
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

"""
writer.flush()
writer.close()
sess.close()
