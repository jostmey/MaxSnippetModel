#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-07-26
# License: See LICENSE
# Purpose: Score statistical classifier on repertoire data on unseen data
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
path_dir = 'PATH_TO_YOUR_VALIDATION_DATA'
samples = dp.load_repertoires(path_dir)
xs, cs, ys = dp.process_repertoires(samples, snip_size=6)

##########################################################################################
# Load parameters
##########################################################################################

# Load parameters from trained model
#
weights_bestfit = np.loadtxt('weights.txt')
weights_bestfit = np.expand_dims(weights_bestfit, 1)
bias_bestfit = np.loadtxt('bias.txt')

##########################################################################################
# Operators
##########################################################################################

# Model settings
#
batch_size = xs.shape[0]
max_instances = xs.shape[1]
num_features = xs.shape[2]

# Inputs
#
features = tf.placeholder(tf.float32, [batch_size, max_instances, num_features])
counts = tf.placeholder(tf.float32, [batch_size, max_instances])
labels = tf.placeholder(tf.float32, [batch_size])

# Repertoire model
#
models = MaxSnippetModel(num_features)
models.weights = tf.Variable(weights_bestfit, name='weights', dtype=tf.float32)
models.biases = tf.Variable(bias_bestfit, name='biases', dtype=tf.float32)
logits = models(features, counts)
cost = models.costs(logits, labels)
accuracy = models.accuracies(logits, labels)

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

  # Evaluate model
  #
  out = session.run(
    (
      cost,
      accuracy
    ),
    feed_dict=feed
  )
  print(
    'Cost:', '%4.3f'%(out[0]/np.log(2.0)),
    'Accuracy:', '%4.3f'%(100.0*out[1])
  )
