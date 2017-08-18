#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-07-26
# License: See LICENSE
# Purpose: Train statistical classifier on repertoire data
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import lib_paths

import dataplumbing as dp
import numpy as np
import tensorflow as tf

from model import *

##########################################################################################
# Data
##########################################################################################

# Load data
#
path_dir = 'PATH_TO_YOUR_TRAINING_DATA'
samples = dp.load_repertoires(path_dir)
xs, cs, ys = dp.process_repertoires(samples, snip_size=6)

##########################################################################################
# Operators
##########################################################################################

# Model settings
#
batch_size = xs.shape[0]
max_instances = xs.shape[1]
num_features = xs.shape[2]

# Training settings
#
learning_rate = 0.01  # Step size
num_iterations = 2500  # Optimization steps
num_replicas = 10000  # Number of attempts to find the best fit

# Inputs
#
features = tf.placeholder(tf.float32, [batch_size, max_instances, num_features])
counts = tf.placeholder(tf.float32, [batch_size, max_instances])
labels = tf.placeholder(tf.float32, [batch_size])

# Repertoire model
#
models = MaxSnippetModel(num_features, num_replicas=num_replicas)
logits, probabilities = models(features, counts)
costs = models.costs(logits, labels)
accuracies = models.accuracies(probabilities, labels)

# Select replica with best fit to the training data
#
index_bestfit = tf.argmin(costs, 0)

# Create operator to optimize the model
#
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(costs)

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

##########################################################################################
# Session
##########################################################################################

# Open session
#
with tf.Session() as session:

  # Initialize variables
  #
  session.run(initializer)

  # Grab a batch of training data
  #
  feed = {features: xs, counts: cs, labels: ys}

  # Each training session represents one batch
  #
  for iteration in range(num_iterations):

    # Optimize model
    #
    out = session.run(
      (
        index_bestfit,
        costs,
        accuracies,
        optimizer
      ),
      feed_dict=feed
    )
    print(
      'Iteration:', iteration,
      'Cost:', '%4.3f'%(out[1][out[0]]/np.log(2.0)),
      'Accuracy:', '%4.3f'%(100.0*out[2][out[0]])
    )

  # Save weights and bias term
  #
  out = session.run(
    (
      index_bestfit,
      models.weights,
      models.biases
    ),
    feed_dict=feed
  )
  weights_bestfit = out[1][:,out[0]]
  bias_bestfit = out[2][out[0]]

##########################################################################################
# Save parameters
##########################################################################################

# Save parameters of trained model
#
np.savetxt('weights.txt', weights_bestfit)
np.savetxt('bias.txt', [bias_bestfit])
