#########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-07-26
# Environment: Python3
# License: See LICENSE
# Purpose: Load dataset and create interfaces for piping the data to the model.
##########################################################################################

import csv
import numpy as np

import lib_paths
import atchley_factors as vector_representation


def load_repertoires(data_dir):
  repertoires = dict()
  with open(data_dir+'/data/answers.csv', 'r') as keyfile_stream:
    keyfile_reader = csv.DictReader(keyfile_stream, delimiter=',')
    for keyfile_row in keyfile_reader:
      sample_id = keyfile_row['Sample']
      diagnosis_id = keyfile_row['Diagnosis']
      sequences = dict()
      path = data_dir+'/data/sample_'+keyfile_row['Sample']+'.csv'
      with open(path, 'r') as sample_stream:
        sample_reader = csv.DictReader(sample_stream, delimiter=',')
        for sample_row in sample_reader:
          sequence = sample_row['Sequence']
          count = float(sample_row['Count'])
          if sequence not in sequences:
            sequences[sequence] = count
          else:
            sequences[sequence] += count
      repertoires[sample_id] = {
        'Diagnosis': diagnosis_id,
        'Sequences': sequences
      }
  return repertoires

def process_repertoires(repertoires, snip_size=6):
  repertoires_snip = {}
  for sample, repertoire in repertoires.items():
    diagnosis = repertoire['Diagnosis']
    snips = {}
    for sequence, count in repertoire['Sequences'].items():
      stop = len(sequence)-snip_size+1
      for i in range(stop):
        snip = sequence[i:i+snip_size]
        if snip not in snips:
          snips[snip] = count
        else:
          snips[snip] += count
    repertoires_snip[sample] = {
      'Diagnosis': diagnosis,
      'Snips': snips
    }

  num_samples = len(repertoires)
  max_snips = -1
  num_features = snip_size*vector_representation.length

  for sample, repertoire in repertoires_snip.items():
    num_snips = len(repertoire['Snips'])
    if num_snips > max_snips:
      max_snips = num_snips

  xs = np.zeros((num_samples, max_snips, num_features), dtype=np.float32)  # Features
  cs = np.zeros((num_samples, max_snips), dtype=np.float32)        # Snippet count
  ys = np.zeros((num_samples), dtype=np.float32)  # Labels

  for i, (sample, repertoire) in enumerate(sorted(repertoires_snip.items(), key=lambda item: item[0])):
    for j, (snip, count) in enumerate(repertoire['Snips'].items()):
      xs[i,j,:] = vector_representation.features(snip)
      cs[i,j] = float(count)
    ys[i] = float(repertoire['Diagnosis'])
  return xs, cs, ys

