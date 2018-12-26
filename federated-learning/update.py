import tensorflow as tf
import h5py
import numpy as np

def onehot(label):
    y = list()

    for i in range(len(label)):
        temp = np.zeros(10, dtype=int)
        temp[label[i]] = 1
        y.append(list(temp))

    y = np.array(y)
    y = np.reshape(y, (len(y), 10))

    return y

dataset = h5py.File('./test.hdf5', 'r')

x = tf.placeholder(tf.float32, [None, 784])

W0 = tf.Variable(tf.zeros([784, 10]))
b0 = tf.Variable(tf.zeros([10]))
W1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.zeros([784, 10]))
b2 = tf.Variable(tf.zeros([10]))

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver({'W': W0, 'b': b0})
saver.restore(sess, './variables_00.ckpt')
saver = tf.train.Saver({'W': W1, 'b': b1})
saver.restore(sess, './variables_01.ckpt')
saver = tf.train.Saver({'W': W2, 'b': b2})
saver.restore(sess, './variables_02.ckpt')

raspi0 = 18400/55000
raspi1 = 18300/55000
raspi2 = 18300/55000

W0 = tf.scalar_mul(raspi0, W0)
W1 = tf.scalar_mul(raspi1, W1)
W2 = tf.scalar_mul(raspi2, W2)
b0 = tf.scalar_mul(raspi0, b0)
b1 = tf.scalar_mul(raspi1, b1)
b2 = tf.scalar_mul(raspi2, b2)

sess.run(W.assign(tf.add(tf.add(W0, W1), W2)))
sess.run(b.assign(tf.add(tf.add(b0, b1), b2)))

saver = tf.train.Saver({'W' : W, 'b' : b})
saver.save(sess, './mean_variables.ckpt')

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Test accuracy:', sess.run(accuracy, feed_dict={x: dataset['test']['input'], y_: onehot(dataset['test']['output'])}))