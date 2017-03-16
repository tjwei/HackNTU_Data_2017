with tf.Session() as sess:
    for op in tf.get_default_graph().get_operations():
        print(op.name, sess.run(op.outputs))