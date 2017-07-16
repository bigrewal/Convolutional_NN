import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

def mnist_model(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1_out = tf.nn.relu(conv2d(x, weight['w1']) + bias['conv_b1'])
    pool1_out = max_pool(conv1_out)

    conv2_out = tf.nn.relu(conv2d(pool1_out, weight['w2']) + bias['conv_b2'])
    pool2_out = max_pool(conv2_out)

    pool2_flat = tf.reshape(pool2_out, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(pool2_flat, weight['w3']) + bias['b3'])


    fc1_drop = tf.nn.dropout(fc1, keep_prob)


    y = tf.matmul(fc1_drop, weight['w4']) + bias['b4']

    return y

saver = tf.train.Saver()

def use_neural_network(digit):
    prediction = mnist_model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./savedModels/mnist_model.ckpt")  #Restoring trained weights and bias

        #Run the model using the trained weights and bias
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [digit],keep_prob: 1.0}), 1)))
        print("Predicted Label: ",result)
        print("Correct Label: ", sess.run(tf.argmax(mnist.test.labels[1], 0)))

use_neural_network(mnist.test.images[1])