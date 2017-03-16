# reduce_sum
tf.reset_default_graph()
one_to_ten = tf.constant(np.arange(1,11), name="one_to_ten")
node = tf.reduce_sum(one_to_ten)
with tf.Session() as sess:
    print("reduce_sum", node.eval())
display(tfdot())

# add_n
tf.reset_default_graph()
node = tf.add_n(list(range(1,11)))
with tf.Session() as sess:
    print("add_n", node.eval())
display(tfdot())

# formula
tf.reset_default_graph()
node10 = tf.constant(10, name='ten')
node = (tf.add(node10,1)*node10)//2
# or
# node = tf.div(tf.multiply(10, tf.add(10,1)), 2)
# or
#node = tf.div(tf.multiply(node10, tf.add(node10,1)), 2)

with tf.Session() as sess:
    print("formula", node.eval())
display(tfdot())

# matmul
tf.reset_default_graph()
ones = tf.ones(shape=(10,1), dtype=tf.float32)
one_to_ten = np.arange(1,11, dtype=np.float32).reshape(1,10)
node = tf.matmul(one_to_ten, ones)
with tf.Session() as sess:
    print("matmul", node.eval())
display(tfdot())

# loop
tf.reset_default_graph()
node = tf.constant(1)
for i in range(2,1+10):
    node = tf.add(node, i)
    # or
    # node = node + i
with tf.Session() as sess:
    print("loop", node.eval())
display(tfdot())

    