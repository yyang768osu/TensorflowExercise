from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Form input/output placeholder
with tf.name_scope('Input'):
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='x-input')
    y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y-input')
    x_image = tf.reshape(x, shape=[-1,28,28,1])

# First convolutional layer
with tf.name_scope('Conv1'):
    W1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32], stddev=0.1), name='W')
    b1 = tf.Variable(tf.constant(value=0.1, shape=[32]), name='B')
    h_conv1 = tf.nn.relu(tf.nn.conv2d(input=x_image,
                                     filter=W1,
                                     strides=[1,1,1,1],
                                     padding='SAME') + b1)
# First pooling layer
with tf.name_scope('Pool1'):
    h_pool1 = tf.nn.max_pool(value=h_conv1,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')

# Second convolutional layer
with tf.name_scope('Conv2'):
    W2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64], stddev=0.1), name='W')
    b2 = tf.Variable(tf.constant(value=0.1, shape=[64]), name='B')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(input=h_pool1,
                                      filter=W2,
                                      strides=[1,1,1,1],
                                      padding='SAME') + b2)

# Second pooling layer
with tf.name_scope('Pool2'):
    h_pool2 = tf.nn.max_pool(value=h_conv2,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')

# Flatten the output of the second pooling layer
with tf.name_scope('Flatten'):
    h_pool2_flat = tf.reshape(tensor=h_pool2, shape=[-1,7*7*64])

# First fully-connected layer
with tf.name_scope('FC1'):
    W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name='W')
    b_fc1 = tf.Variable(tf.constant(value=0.1, shape=[1024]), name='B')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# Drop out after first fully-connected layer
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# Second fully-connected layer
with tf.name_scope('FC2'):
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1), name='W')
    b_fc2 = tf.Variable(tf.constant(value=0.1, shape=[10]), name='B')
    y_conv = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2)+b_fc2)

# cross entropy
with tf.name_scope('CrossEntropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
    )

# train step
with tf.name_scope('Training'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# calculate accuracy
with tf.name_scope('CalcAccuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1), name='EqualOrNot')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')

# graph to file
writer = tf.summary.FileWriter("./temp/")
with tf.Session() as sess:
    writer.add_graph(sess.graph)

# add summaries
tf.summary.scalar('cross-entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 10 == 0:
            summary = sess.run(merged_summary, feed_dict={x: batch[0],
                                                          y: batch[1],
                                                          keep_prob: 1.0
                                                          })
            writer.add_summary(summary, i)

            train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                      y: batch[1],
                                                      keep_prob: 1.0
                                                      })
            print("step: {}, accuracy: {}".format(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0],
                                  y: batch[1],
                                  keep_prob: 0.5
                                  })
    test_accuracy = accuracy.eval(feed_dict={
        x: mnist.test.images,
        y: mnist.test.labels,
        keep_prob: 1.0
    })
    print("Test accuracy: {}".format(test_accuracy))    
