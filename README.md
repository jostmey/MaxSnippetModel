# Statistical classifiers for diagnosing disease from immune repertoires
###### LABORATORY OF DR. LINDSAY COWELL

## Description

The full set of antibodies and immune receptors in an individual contains traces of past and current immune responses. These traces can serve as biomarkers for diseases mediated by the adaptive immune system (e.g. infectious disease, organ rejection, autoimmune disease, cancer). Only a handful of immune receptors that can be sequenced from a patient are expected to contain these traces. Here we present the source code to a method for elucidating these traces.

First, the CDR3 is parsed from every antibody sequence in a patient (see [VDJ Server](https://vdjserver.org/)). The CDR3 is then cut into fixed-length subsequences that we call snippets. These are nothing more than the k-mers of the CDR3. The amino acid residues of each snippet are then described by their biochemical properties in a position dependent manner using [Atchley factors](http://www.pnas.org/content/102/18/6395.full).

The main idea is to score every snippet by its biochemical features with a dectector function and to aggregate the scores into a single value that can represent a diagnosis. Because only a handful of snippets are expected to have a high score in patients with a disease, we aggregate the scores together by taking the maximum score. The maximum score is then used to predict the probability that a patient has a positive diagnosis (a high score would suggest a positive diagnosis, no high scores would suggest a negative diagnosis). The parameters of the detector function are fitted by maximizing the log-likelihood (minimizing the cross-entropy error) that each diagnosis is correct.

The model is fitted to the training data using gradient based optimization techniques. First, initial values are randomly drawn for each parameter. Then 2,500 steps of gradient based optimization are used to find a locally optimal fit to the data. We find that the fitting procedure must be repeated hundreds of thousands of times to find a good fit to the training data.

For a complete description of this approach, see our publication in BMC Bioinformatics:

 * [Statistical classifiers for diagnosing disease from immune repertoires: a case study using multiple sclerosis](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1814-6)

## Requirements

 * [Python3](https://www.python.org/)
 * [TensorFlow](https://www.tensorflow.org/)
 * [NumPy](http://www.numpy.org/)

## Download

 * Download: [zip](https://github.com/jostmey/MaxSnippetModel/zipball/master)
 * Git: `git clone https://github.com/jostmey/MaxSnippetModel`

## Primary Files

 * model.py
 * train.py
 * score.py
 * Data used to develop the approach cannot be made available at this time
