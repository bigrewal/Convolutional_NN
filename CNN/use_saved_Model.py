import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import CNN_mnist

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

#Placeholders for our data
x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.float32, shape=(None, 10))
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#Lets define our weights
weight= {
    'w1': tf.Variable(tf.random_normal([5,5,1,32], stddev=0.35),name="w1"),
    'w2': tf.Variable(tf.random_normal([5,5,32,64], stddev=0.35),name="w2"),
    'w3': tf.Variable(tf.random_normal([7*7*64,1024], stddev=0.35),name="w3"),
    'w4': tf.Variable(tf.random_normal([1024,10], stddev=0.35),name="w4")
}

#Lets define our biases
bias={
    'conv_b1' : tf.Variable(tf.zeros([32]), name="conv_b1"),
    'conv_b2' : tf.Variable(tf.zeros([64]), name="conv_b2"),
    'b3': tf.Variable(tf.zeros([1024]), name="b3"),
    'b4': tf.Variable(tf.zeros([n_classes]), name="b4")
}

#CNN_mnist.my_model()
saver = tf.train.Saver()

def use_neural_network(digit):
    prediction = CNN_mnist.my_model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./savedModels/mnist_model.ckpt")

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [digit],keep_prob: 1.0}), 1)))
        print("Predicted Label: ",result)
        print("Correct Label: ", sess.run(tf.argmax(mnist.test.labels[1], 0)))

use_neural_network(mnist.test.images[1])