import tensorflow as tf
import numpy as np

batch_size = 3


def generator(is_training, batch_size, z):
    with tf.variable_scope('generator'):
        with tf.variable_scope('linear1'):
            W = tf.get_variable(name='weight', shape=[10, 1024],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[1024],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.matmul(z, W) + b
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,  # set to false, since activation is relu
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = tf.nn.relu(net)

        with tf.variable_scope('linear2'):
            W = tf.get_variable(name='weight', shape=[1024, 7 * 7 * 128],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[7 * 7 * 128],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.matmul(net, W) + b
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,  # set to false, since activation is relu
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = tf.nn.relu(net)
            net = tf.reshape(
                net, shape=[batch_size, 7, 7, 128], name='Flattened')

        with tf.variable_scope('convTranspose1'):
            W = tf.get_variable(name='weight', shape=[4, 4, 64, 128],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[64],
                                initializer=tf.constant_initializer(value=0.1))
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
                                               is_training=is_training)
            net = tf.nn.relu(net)

        with tf.variable_scope('convTranspose2'):
            W = tf.get_variable(name='weight', shape=[4, 4, 1, 64],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[1],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.nn.conv2d_transpose(
                value=net,
                filter=W,
                output_shape=[batch_size, 28, 28, 1],
                strides=[1, 2, 2, 1],
                padding='SAME'
            ) + b
            net = tf.nn.relu(net)

        net = tf.nn.sigmoid(net)
        tf.summary.image('Image', net, max_outputs=50)
    return net


def discriminator(is_training, batch_size, x):
    with tf.variable_scope('discriminator'):
        with tf.variable_scope('conv1'):
            W = tf.get_variable(name='weight', shape=[4, 4, 1, 64],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[64],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.nn.conv2d(
                input=x,
                filter=W,
                strides=[1, 2, 2, 1],
                padding='SAME',
            )
            net = tf.nn.relu(net)

        with tf.variable_scope('conv2'):
            W = tf.get_variable(name='weight', shape=[4, 4, 64, 128],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[128],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.nn.conv2d(
                input=net,
                filter=W,
                strides=[1, 2, 2, 1],
                padding='SAME'
            )
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,  # set to false, since activation is relu
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = tf.nn.relu(net)

        with tf.variable_scope('linear1'):
            net = tf.reshape(net, shape=[batch_size, 7 * 7 * 128])
            W = tf.get_variable(name='weight', shape=[7 * 7 * 128, 1024],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[1024],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.matmul(net, W) + b
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,  # set to false, since activation is relu
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = tf.nn.relu(net)

        with tf.variable_scope('linear2'):
            W = tf.get_variable(name='weight', shape=[1024, 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[1],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.matmul(net, W) + b
        net = tf.sigmoid(net)
    return net


z = tf.placeholder(shape=[batch_size, 10],
                   dtype=tf.float32, name='z')
x = tf.placeholder(shape=[batch_size, 28, 28, 1],
                   dtype=tf.float32, name='x')

generator_net = generator(is_training=True, batch_size=batch_size, z=z)
discriminator_net = discriminator(is_training=True, batch_size=batch_size, x=x)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./temp/")

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        z_samples = np.random.uniform(-1, 1, size=(batch_size, 10))
        generator_net.eval(feed_dict={z: z_samples})
        summary = sess.run(merged_summary, feed_dict={z: z_samples})
        writer.add_summary(summary, i)


writer.close()
