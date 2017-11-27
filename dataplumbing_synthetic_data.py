#########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-07-26
# Environment: Python3
# License: See LICENSE
# Purpose: Generate synthetic dataset and create interfaces for piping the data to the model.
# Note: Overwrite "dataplumbing.py" with this file to use synthetic data.
##########################################################################################

import numpy as np

import lib_paths
import atchley_factors as vector_representation


def load_repertoires(data_dir):
  return None

def process_repertoires(repertoires, snip_size=6):

  # NOTE:
  #
  # This script creates a set of random of snippets (k-mers) for each sample.
  # If this were real data from immune receptor sequences, you would take each
  # CDR3 sequence in a sample, cut it up into every possible snippet (k-mer),
  # and use those snippets (see EXAMPLE below). The snippet count would be the total number of
  # times the snippet appeared in all of the CDR3 sequences from a sample.
  #
  # EXAMPLE:
  #
  # Assume this is your CDR3:
  #  ACTRGHKCILR
  # The snippets are:
  #  ACTRGH     
  #   CTRGH     
  #    TRGHKC   
  #     RGHKCI  
  #      GHKCIL 
  #       HKCILR
  # This must be done for every CDR3 in the sample. After conveting the snippets
  # into a vector representation (Atchley factors), the values are stored in "xs".

  # Data dimensions
  #
  num_samples = 20
  num_snips_per_sample = 300
  snip_size = 6
  num_features = snip_size*vector_representation.length

  # Data variables
  #
  xs = np.zeros((num_samples, num_snips_per_sample, num_features), dtype=np.float32)	# Features
  cs = np.zeros((num_samples, num_snips_per_sample), dtype=np.float32)	# Snippet count
  ys = np.zeros((num_samples), dtype=np.float32)	# Labels

  # Generate random snippets
  #
  aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
  for i in range(num_samples):
    N = np.random.randint(round(num_snips_per_sample/2))+round(num_snips_per_sample/2)-1
    for j in range(N):
      snip = ''
      for k in range(snip_size):
        index = np.random.randint(len(aa_list))
        snip += aa_list[index]
      xs[i,j,:] = vector_representation.features(snip)
      cs[i,j] = 1.0

  # Place needle in some samples and give those samples a positive diagnosis
  #
  needle = 'ARKIHG'
  for i in range(round(num_samples/2)):
    ys[i] = 1.0
    xs[i,0,:] = vector_representation.features(needle)

  return xs, cs, ys

