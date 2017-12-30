import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Hyper-parameter in https://arxiv.org/abs/1511.06434:
# leaky_relu with alpha = 0.2
# BN used in all layers except for input and output

# Hyper-prarameter in https://arxiv.org/pdf/1701.07875:
# learning_rate = 0.00005
# batch_size = 64
# clip_value = 0.01
# n_critic = 5

batch_size = 64
leaky_relu_alpha = 0.2
learning_rate = 0.00005
clip_value = 0.01
n_critic = 5


def leaky_relu(x, alpha=leaky_relu_alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def generator(is_training, batch_size, random_vector, reuse=True):
    with tf.variable_scope('generator', reuse=reuse):
        with tf.variable_scope('linear1'):
            W = tf.get_variable(name='weight', shape=[10, 1024],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[1024],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.matmul(random_vector, W) + b
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = leaky_relu(net)

        with tf.variable_scope('linear2'):
            W = tf.get_variable(name='weight', shape=[1024, 7 * 7 * 128],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='bias', shape=[7 * 7 * 128],
                                initializer=tf.constant_initializer(value=0.1))
            net = tf.matmul(net, W) + b
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = leaky_relu(net)
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
                                               scale=False,
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = leaky_relu(net)

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
            net = leaky_relu(net)

        net = tf.nn.sigmoid(net)
    return net


def discriminator(is_training, batch_size, image_input, reuse=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.variable_scope('conv1'):
            W = tf.get_variable(name='weight', shape=[4, 4, 1, 64],
                                initializer=tf.truncated_normal_initializer(stddev=clip_value/2))
            b = tf.get_variable(name='bias', shape=[64],
                                initializer=tf.constant_initializer(value=0))
            net = tf.nn.conv2d(
                input=image_input,
                filter=W,
                strides=[1, 2, 2, 1],
                padding='SAME',
            )
            net = leaky_relu(net)

        with tf.variable_scope('conv2'):
            W = tf.get_variable(name='weight', shape=[4, 4, 64, 128],
                                initializer=tf.truncated_normal_initializer(stddev=clip_value/2))
            b = tf.get_variable(name='bias', shape=[128],
                                initializer=tf.constant_initializer(value=0))
            net = tf.nn.conv2d(
                input=net,
                filter=W,
                strides=[1, 2, 2, 1],
                padding='SAME'
            )
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = leaky_relu(net)

        with tf.variable_scope('linear1'):
            net = tf.reshape(net, shape=[batch_size, 7 * 7 * 128])
            W = tf.get_variable(name='weight', shape=[7 * 7 * 128, 1024],
                                initializer=tf.truncated_normal_initializer(stddev=clip_value/2))
            b = tf.get_variable(name='bias', shape=[1024],
                                initializer=tf.constant_initializer(value=0))
            net = tf.matmul(net, W) + b
            net = tf.contrib.layers.batch_norm(inputs=net,
                                               decay=0.9,
                                               center=True,  # allow beta to be updated
                                               scale=False,
                                               epsilon=0.001,
                                               updates_collections=None,
                                               is_training=is_training)
            net = leaky_relu(net)

        with tf.variable_scope('linear2'):
            W = tf.get_variable(name='weight', shape=[1024, 1],
                                initializer=tf.truncated_normal_initializer(stddev=clip_value/2))
            b = tf.get_variable(name='bias', shape=[1],
                                initializer=tf.constant_initializer(value=0))
            net = tf.matmul(net, W) + b
    return net


def clipping(var_list):
    with tf.name_scope('clipping'):
        return [var.assign(tf.clip_by_value(var, -clip_value, clip_value)) for var in var_list]


random_vector = tf.placeholder(shape=[batch_size, 10],
                               dtype=tf.float32, name='random_vector')
# random_vector_for_testing = tf.placeholder(shape=[1, 10],
#                                           dtype = tf.float32, name = 'random_vector')
true_image=tf.placeholder(shape = [batch_size, 28, 28, 1],
                            dtype=tf.float32, name='true_image')

fake_image_for_training = generator(
    is_training=True, batch_size=batch_size, random_vector=random_vector, reuse=False)
# fake_image_for_testing = generator(
#    is_training=False, batch_size=1, random_vector=random_vector_for_testing, reuse=True)
discriminator_output_true = discriminator(
    is_training=True, batch_size=batch_size, image_input=true_image, reuse=False)
discriminator_output_fake = discriminator(
    is_training=True, batch_size=batch_size, image_input=fake_image_for_training, reuse=True)

inner_loop_min_goal = tf.reduce_mean(discriminator_output_fake) - tf.reduce_mean(discriminator_output_true)
outer_loop_min_goal = - tf.reduce_mean(discriminator_output_fake)
fake_image_summary = tf.summary.image('FakeImage', fake_image_for_training, max_outputs=64)
true_image_summary = tf.summary.image('TrueImage', true_image, max_outputs=64)


discriminator_mean_fake_output = tf.summary.scalar("discriminator_mean_fake_output", tf.reduce_mean(discriminator_output_fake))
discriminator_mean_true_output = tf.summary.scalar("discriminator_mean_true_output", tf.reduce_mean(discriminator_output_true))
discriminator_inner_loop_loss = tf.summary.scalar("discriminator_inner_loop_loss", inner_loop_min_goal)
discriminator_outer_loop_loss = tf.summary.scalar("discriminator_outer_loop_loss", outer_loop_min_goal)

inner_loop_summary = tf.summary.merge([discriminator_mean_fake_output, discriminator_mean_true_output, discriminator_inner_loop_loss])
outer_loop_summary = tf.summary.merge([discriminator_outer_loop_loss, fake_image_summary, true_image_summary])

generator_trainable_var = [
    var for var in tf.trainable_variables() if var.name.startswith('generator')]
discriminator_trainable_var = [
    var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

clip_discriminator = clipping(discriminator_trainable_var)

inner_loop_trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(
    loss=inner_loop_min_goal, var_list=discriminator_trainable_var)
outer_loop_trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(
    loss=outer_loop_min_goal, var_list=generator_trainable_var)


print('trainable variable in generator:')
for var in generator_trainable_var:
    print(var.name)
print('trainable variable in discriminator:')
for var in discriminator_trainable_var:
    print(var.name)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./temp/")

with tf.Session() as sess:
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(clip_discriminator)
    counter = 0
    while mnist.train.epochs_completed < 5:
        print('Epoch idx: {}, Sample idx: {}'.format(mnist.train.epochs_completed, mnist.train._index_in_epoch))

        random_vector_samples = np.random.uniform(-1, 1, size=(batch_size, 10))
        true_image_samples = mnist.train.next_batch(batch_size)[0].reshape(batch_size, 28, 28, 1)
        [_, _, summary] = sess.run([inner_loop_trainer, clip_discriminator, inner_loop_summary], 
                                    feed_dict={random_vector: random_vector_samples,
                                               true_image: true_image_samples})
        writer.add_summary(summary, counter)
        
        # Outer loop
        if counter % n_critic == 0:
            random_vector_samples = np.random.uniform(-1, 1, size=(batch_size, 10))
            [_, summary] = sess.run([outer_loop_trainer, outer_loop_summary], 
                                    feed_dict={random_vector: random_vector_samples,
                                                true_image: true_image_samples})
            writer.add_summary(summary, counter)
        
        counter += 1


writer.close()
