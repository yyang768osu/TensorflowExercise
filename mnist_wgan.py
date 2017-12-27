import tensorflow as tf
import numpy as np

batch_size = 3

with tf.name_scope('InputRandomNoise'):
    z = tf.placeholder(shape=[batch_size, 10], dtype=tf.float32, name='RandomZ')

with tf.name_scope('Linear1'):
    W = tf.Variable(tf.truncated_normal(
        shape=[10, 1024], stddev=0.1), name='Weight')
    b = tf.Variable(tf.constant(0.1, shape=[1024]), name='Bias')
    net = tf.matmul(z, W) + b
    net = tf.contrib.layers.batch_norm(inputs=net,
                                       decay=0.9,
                                       center=True,  # allow beta to be updated
                                       scale=False,  # set to false, since activation is relu
                                       epsilon=0.001,
                                       updates_collections=None,
                                       is_training=True)
    net = tf.nn.relu(net)

with tf.name_scope('Linear2'):
    W = tf.Variable(tf.truncated_normal(
        shape=[1024, 7 * 7 * 128], stddev=0.1), name='Weight')
    b = tf.Variable(tf.constant(0.1, shape=[7 * 7 * 128]), name='Bias')
    net = tf.matmul(net, W) + b
    net = tf.contrib.layers.batch_norm(inputs=net,
                                       decay=0.9,
                                       center=True,  # allow beta to be updated
                                       scale=False,  # set to false, since activation is relu
                                       epsilon=0.001,
                                       updates_collections=None,
                                       is_training=True)
    net = tf.nn.relu(net)
    net = tf.reshape(net, shape=[batch_size, 7, 7, 128], name='Flattened')

with tf.name_scope('ConvTranspose1'):
    W = tf.Variable(tf.truncated_normal(
        shape=[4, 4, 64, 128], stddev=0.1), name='Weight')
    b = tf.Variable(tf.constant(0.1, shape=[64]))
    net = tf.nn.conv2d_transpose(
        value=net,
        filter=W,
        output_shape=[batch_size, 14, 14, 64],
        strides=[1, 2, 2, 1],
        padding='SAME'
    ) + b
    net = tf.contrib.layers.batch_norm(inputs=net,
                                       decay=0.9,
                                       center=True,  # allow beta to be updated
                                       scale=False,  # set to false, since activation is relu
                                       epsilon=0.001,
                                       updates_collections=None,
                                       is_training=True)
    net = tf.nn.relu(net)

with tf.name_scope('ConvTranspose2'):
    W = tf.Variable(tf.truncated_normal(
        shape=[4, 4, 1, 64], stddev=0.1), name='Weight')
    b = tf.Variable(tf.constant(0.1, shape=[1]))
    net = tf.nn.conv2d_transpose(
        value=net,
        filter=W,
        output_shape=[batch_size, 28, 28, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    ) + b
    net = tf.nn.relu(net)
    tf.summary.image('Image', net, max_outputs=50)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./temp/")

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        z_samples = np.random.uniform(-1, 1, size=(batch_size, 10))
        net.eval(feed_dict={z:z_samples})
        summary = sess.run(merged_summary, feed_dict={z: z_samples})
        writer.add_summary(summary, i)


writer.close()
