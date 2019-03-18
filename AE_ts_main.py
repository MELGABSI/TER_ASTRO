# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""

import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector
from AE_ts_model import Model, open_data, plot_data, plot_z_run
tf.logging.set_verbosity(tf.logging.ERROR)


"""Hyperparameters"""

LOG_DIR = "./Saved_Model"  # Directory for the logging

config = dict()  # Put all configuration information into the dict
config['num_layers'] = 2  # number of layers of stacked RNN's
config['hidden_size'] = 90  # memory cells in a layer
config['max_grad_norm'] = 0.5  # maximum gradient norm during training
config['batch_size'] = batch_size = 64
config['learning_rate'] = .005
config['num_l'] = 20  # number of units in the latent space

plot_every = 100  # after _plot_every_ GD steps, there's console output
max_iterations = 100  # maximum number of iterations
dropout = 0.8  # Dropout rate


# Load the data
X_train, X_val = open_data('./data/')

N = X_train.shape[0] # nbr of element in train db
Nval = X_val.shape[0] # nbr of element in val db
D = X_train.shape[1] # nbr of columns in train db
config['sl'] = sl = D  # sequence length
print('We have %s observations with %s dimensions' % (N, D))

# Organize the classes
num_classes = 15



"""Training time!"""
model = Model(config)
sess = tf.Session()
perf_collect = np.zeros((2, int(np.floor(max_iterations / plot_every)))) # np.floor -> a = np.array([-1.7,2.0]) np.floor(a) array([-2.,  2.]) *** np.zeros Return a new array of given shape filled with zeros.

do_train = True
if do_train:

    # Proclaim the epochs
    epochs = np.floor(batch_size * max_iterations / N)
    print('Train with approximately %d epochs' % epochs)

    sess.run(model.init_op)
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  # writer for Tensorboard

    step = 0  # Step is a counter for filling the numpy array perf_collect
    for i in range(max_iterations):
        batch_ind = np.random.choice(N, batch_size, replace=False)
        result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.train_step],
                          feed_dict={model.x: X_train[batch_ind], model.keep_prob: dropout})

        if i % plot_every == 0:
            # Save train performances
            perf_collect[0, step] = loss_train = result[0]
            loss_train_seq, lost_train_lat = result[1], result[2]

            # Calculate and save validation performance
            batch_ind_val = np.random.choice(Nval, batch_size, replace=False)

            result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.merged],
                              feed_dict={model.x: X_val[batch_ind_val], model.keep_prob: 1.0})
            perf_collect[1, step] = loss_val = result[0]
            loss_val_seq, lost_val_lat = result[1], result[2]
            # and save to Tensorboard
            summary_str = result[3]
            writer.add_summary(summary_str, i)
            writer.flush()

            print("At %6s / %6s train (%5.3f, %5.3f, %5.3f), val (%5.3f, %5.3f,%5.3f) in order (total, seq, lat)" % (
            i, max_iterations, loss_train, loss_train_seq, lost_train_lat, loss_val, loss_val_seq, lost_val_lat))
            step += 1

    # Save the model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)



do_plot = True
if do_plot:
    # Extract the latent space coordinates of the validation set
    start = 0
    label = []  # The label to save to visualize the latent space
    z_run = []

    while start + batch_size < Nval:
        run_ind = range(start, start + batch_size)
        z_mu_fetch = sess.run(model.z_mu, feed_dict={model.x: X_val[run_ind], model.keep_prob: 1.0})
        z_run.append(z_mu_fetch)
        start += batch_size

    z_run = np.concatenate(z_run, axis=0)
    label = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    plot_z_run(z_run, label)

sess.close()
# Now open Tensorboard with
#  $tensorboard --logdir = LOG_DIR
