#!/usr/bin/env python3
"""Builds, trains, and saves a deep network with Adam, mini-batch GD,
inverse time decay, and batch normalization."""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_placeholders(nx, classes):
    """Return placeholders x and y for input data and labels."""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    """Return a dense layer's activated output without batch norm."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init
    )
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """Return the activated output of a dense + batch norm layer."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.ones((n,)), name='gamma')
    beta = tf.Variable(tf.zeros((n,)), name='beta')
    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, 1e-8)
    if activation is None:
        return z_norm
    return activation(z_norm)


def forward_prop(x, layer_sizes, activations):
    """Return the prediction tensor; batch norm all hidden layers."""
    a = x
    last = len(layer_sizes) - 1
    for i in range(len(layer_sizes)):
        if i == last:
            a = create_layer(a, layer_sizes[i], activations[i])
        else:
            a = create_batch_norm_layer(a, layer_sizes[i], activations[i])
    return a


def calculate_accuracy(y, y_pred):
    """Return the decimal accuracy tensor."""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def calculate_loss(y, y_pred):
    """Return the softmax cross-entropy loss tensor."""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Build, train, and save the network; return the save path."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    decayed_alpha = tf.train.inverse_time_decay(
        alpha, global_step, 1, decay_rate, staircase=True
    )
    train_op = tf.train.AdamOptimizer(
        decayed_alpha, beta1, beta2, epsilon
    ).minimize(loss)
    increment_step = tf.assign(global_step, global_step + 1)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        for epoch in range(epochs + 1):
            t_cost, t_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train}
            )
            v_cost, v_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_acc))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_acc))

            if epoch == epochs:
                break

            X_sh, Y_sh = shuffle_data(X_train, Y_train)
            step = 0
            for start in range(0, m, batch_size):
                end = start + batch_size
                if end > m:
                    end = m
                Xb = X_sh[start:end]
                Yb = Y_sh[start:end]
                sess.run(train_op, feed_dict={x: Xb, y: Yb})
                step += 1
                if step % 100 == 0:
                    s_cost, s_acc = sess.run(
                        [loss, accuracy], feed_dict={x: Xb, y: Yb}
                    )
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(s_cost))
                    print("\t\tAccuracy: {}".format(s_acc))
            sess.run(increment_step)

        return saver.save(sess, save_path)
