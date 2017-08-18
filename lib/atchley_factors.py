##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-05-02
# Environment: Python3
# License: See LICENSE
# Purpose: Tools for describing an amino acid sequence as a sequence of Atchley factors.
##########################################################################################

__path = '/'.join(__file__.split('/')[:-1])+'/atchley_factors.csv'

vecs = dict()
with open(__path, 'r') as stream:
	for line in stream:
		row = line.split(',')
		key = row[0]
		values = []
		for value in row[1:]:
			values.append(float(value))
		vecs[key] = values
length = len(vecs['A'])
labels = ['I', 'II', 'III', 'IV', 'V']

def features(sequence):
	values = []
	for aa in sequence:
		values += vecs[aa]
	return values

