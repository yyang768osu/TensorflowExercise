from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)

x_image = tf.reshape(x, shape=[-1,28,28,1])

W1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[32]))

h_conv1 = tf.nn.relu(tf.nn.conv2d(input=x_image,
                                  filter=W1,
                                  strides=[1,1,1,1],
                                  padding='SAME') + b1)
h_pool1 = tf.nn.max_pool(value=h_conv1,
                         ksize=[1,2,2,1],
                         strides=[1,2,2,1],
                         padding='SAME')

W2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(input=h_pool1,
                                  filter=W2,
                                  strides=[1,1,1,1],
                                  padding='SAME') + b2)
h_pool2 = tf.nn.max_pool(value=h_conv2,
                         ksize=[1,2,2,1],
                         strides=[1,2,2,1],
                         padding='SAME')

h_pool2_flat = tf.reshape(tensor=h_pool2, shape=[-1,7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2)+b_fc2)

# cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
)

# train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        batch = mnist.train.next_batch(50)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0],
                y: batch[1],
                keep_prob: 1.0
            })
            print("step: {}, accuracy: {}".format(i, train_accuracy))
        train_step.run(feed_dict={
            x: batch[0],
            y: batch[1],
            keep_prob: 0.5
        })
    test_accuracy = accuracy.eval(feed_dict={
        x: mnist.test.images,
        y: mnist.test.labels,
        keep_prob: 1.0
    })
    print("Test accuracy: {}".format(test_accuracy))    
    
    