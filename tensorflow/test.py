import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = tf.multiply(a, b)

# lazy
sess = tf.Session()

print(sess.run(y, feed_dict={ a: 3, b: 3 }))