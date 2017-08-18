# Statistical Classifier for Diagnosing Disease from Immune Repertoires

## Description

With the advent of high throughput immune receptor sequencing, a sample of a patient's immune repertoire can now be affordably sequenced. The challenge now is to find biomarkers in the individual immune receptor sequences. We have taken a first step toward this goal building a statistical classifier that maps immune repertoires to patient diagnoses.

The model consists of two parts. The first part is a detector function. The detector function scores the features of each snippet of receptor sequence. For the detector function, we decided to use the same model that is used in logistic regression. Once every snippet of receptor sequence has been scored, the second part of the model aggregates the scores into a single diagnosis. Any [generalized mean](https://en.wikipedia.org/wiki/Generalized_mean) is formally appropriate for aggregating the scores. For this project, we decided to use the *max mean*. The  max mean aggregates a set of scores by taking the highest score. This makes sense if the expected signal is sparse.

Once the model is defined, it needs to be fitted to a set of training data. Each item of training data consists of a patient's immune receptor repertoire and their associated diagnosis. The scoring function can then be fitted to the data by finding the model parameters that maximize the likelihood that each diagnoses is correct.

In this project, gradient based optimization techniques are used to fit the model parameters to the data. Because gradient based optimization techniques will find a local best fit to the data, the fitting procedure must be repeated many times. Each time the fitting procedure is repeated, the model parameters are randomly initialized at a different starting location.

## Code

The code is written in [TensorFlow](https://www.tensorflow.org/), an open source machine learning package published by Google. Using TensorFlow, the gradient calculations are completely automated, removing the need to work out by hand the derivatives of the scoring function and aggregation method. TensorFlow also makes it easy to score each immune receptor in parallel.

Using TensorFlow, we setup thousands of duplicate models to run in parallel. Each model contains a different initial guess for the model parameters. The models are fitted to the training data in parallel, and the best fitting model corresponds to the local optimum that best explains the training data.

The script is located in `train.py`.

## Requirements

 * [Python3](https://www.python.org/)
 * [TensorFlow](https://www.tensorflow.org/)
 * [NumPy](http://www.numpy.org/)

## Download

* Download: [zip](https://github.com/jostmey/rwa/zipball/master)
* Git: `git clone https://github.com/jostmey/rwa`
