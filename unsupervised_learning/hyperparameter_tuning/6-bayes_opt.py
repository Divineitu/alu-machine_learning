#!/usr/bin/env python3
"""Bayesian hyperparameter tuning of a Keras MNIST classifier with GPyOpt."""

import os

import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras


def load_data():
    """Return the MNIST data as flattened, normalized arrays."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train_oh, x_test, y_test_oh


X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = load_data()


def build_and_train(params):
    """Train a model with the given hyperparameters and return val loss."""
    lr = float(params[0, 0])
    units = int(params[0, 1])
    dropout = float(params[0, 2])
    l2 = float(params[0, 3])
    batch_size = int(params[0, 4])

    model = keras.Sequential([
        keras.layers.Dense(
            units, activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2),
            input_shape=(784,)
        ),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    ckpt_name = (
        "ckpt_lr-{:.5f}_units-{}_drop-{:.2f}"
        "_l2-{:.5f}_bs-{}.h5"
    ).format(lr, units, dropout, l2, batch_size)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_name, monitor='val_loss', save_best_only=True
        ),
    ]

    history = model.fit(
        X_TRAIN, Y_TRAIN,
        batch_size=batch_size,
        epochs=30,
        validation_data=(X_VAL, Y_VAL),
        callbacks=callbacks,
        verbose=0,
    )
    return float(np.min(history.history['val_loss']))


BOUNDS = [
    {'name': 'lr', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'units', 'type': 'discrete',
     'domain': (32, 64, 128, 256, 512)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (16, 32, 64, 128)},
]


if __name__ == '__main__':
    opt = GPyOpt.methods.BayesianOptimization(
        f=build_and_train,
        domain=BOUNDS,
        acquisition_type='EI',
        maximize=False,
    )
    opt.run_optimization(max_iter=30)

    opt.plot_convergence()
    plt.savefig('convergence.png')

    with open('bayes_opt.txt', 'w') as report:
        report.write('Bayesian Optimization Report\n')
        report.write('=' * 32 + '\n\n')
        report.write('Best hyperparameters:\n')
        names = [b['name'] for b in BOUNDS]
        for name, value in zip(names, opt.x_opt):
            report.write('  {}: {}\n'.format(name, value))
        report.write('\nBest validation loss: {}\n'.format(opt.fx_opt))
        report.write(
            '\nCheckpoint files for each evaluated configuration are '
            'saved in the working directory.\n'
        )
