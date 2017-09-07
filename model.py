##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started2016-07-26
# License: See LICENSE
# Purpose: Statistical model for classifying repertoire data
##########################################################################################

import tensorflow as tf


class MaxSnippetModel:
  """Statistical model for classifying repertoire data"""

  def __init__(self, num_features, num_replicas=1):
    """
    Args:
      num_features: int, Number of features in each instance.
      num_replicas (optional): int, Number of copies of the model to fit the training
        data. Each replica will start with different initial guesses
        for the parameters.

    """

    # Number of replicas
    #
    self.num_replicas = num_replicas

    # Initialize parameters for models
    #
    self.weights = tf.get_variable(
      'weights', [num_features, num_replicas],
      initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True)  # Randomly initialize weights
    )
    self.biases = tf.get_variable(
      'bias', [num_replicas],
      initializer=tf.constant_initializer(0.0)
    )

  def __call__(self, features, counts):
    """
    Args:
      features: `3D` Tensor, the features for each instance of every sample.
      counts `2D` Tensor, the number of times each instance appears in a sample.
    Returns:
      - `2D` Tensor, logits of the sigmoid function. The first dimension covers
        every sample. The second dimension covers every replica.
      - `2D` Tensor, probability of positive diagnosis. The first dimension covers
        every sample. The second dimension covers every replica.
    """

    # Settings
    #
    shape = features.get_shape()
    batch_size = int(shape[0])
    max_instances = int(shape[1])
    num_features = int(shape[2])

    # Score features
    #
    features_flat = tf.reshape(features, [batch_size*max_instances, num_features])
    scores_flat = tf.matmul(features_flat, self.weights)+self.biases
    scores = tf.reshape(scores_flat, [batch_size, max_instances, self.num_replicas])

    # Aggregate scores
    #
    counts_expand = tf.expand_dims(counts, axis=2)
    penalties = -1E12*(1.0-tf.sign(counts_expand))  # No penalty if counts > 0. The penalty is -1E12 when counts are zero.
    logits = tf.reduce_max(scores+penalties, axis=1)

    return logits

  def costs(self, logits, labels):
    """
    Args:
      logits: `2D` Tensor, logits of the sigmoid function. The first dimension
        covers every sample. The second dimension covers every replica.
      labels `1D` Tensor, the label for each sample.
    Returns:
      - `1D` Tensor, error between the model and labels across every replica.
    """

    # Tile inputs over every replica
    #
    labels_expand = tf.expand_dims(labels, axis=1)
    labels_tile = tf.tile(labels_expand, [1, self.num_replicas])

    # Cost function
    #
    costs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_tile), axis=0)

    return costs

  def accuracies(self, logits, labels):
    """
    Args:
      probabilities: `2D` Tensor, logits of the sigmoid function. The first dimension
        covers every sample. The second dimension covers every replica.
      labels `1D` Tensor, the label for each sample.
    Returns:
      - `1D` Tensor, fraction of times the model prediction is correct for every
        replica.
    """

    # Compute probabilities
    #
    probabilities = tf.sigmoid(logits)

    # Tile inputs over every replica
    #
    labels_expand = tf.expand_dims(labels, axis=1)
    labels_tile = tf.tile(labels_expand, [1, self.num_replicas])

    # Accuracy function
    #
    correct = tf.equal(tf.round(labels_tile), tf.round(probabilities))
    accuracies = tf.reduce_mean(tf.cast(correct, tf.float32), axis=0)

    return accuracies

