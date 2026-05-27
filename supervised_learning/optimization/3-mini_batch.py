#!/usr/bin/env python3
"""Trains a loaded neural network using mini-batch gradient descent."""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Train a restored model with mini-batches and save it."""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

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

        return saver.save(sess, save_path)
