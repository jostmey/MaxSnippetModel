##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-05-02
# Environment: Python3
# License: See LICENSE
# Purpose: Tools for representing a nucleotide sequence as a sequence of categorical variables.
##########################################################################################

length = 4
labels = ['A', 'C', 'G', 'T']

def features(sequence):
	values = [0.0]*(length*len(sequence))
	for index, nn in enumerate(sequence):
		if nn is 'A': values[length*index] = 1.0
		elif nn is 'C':values[length*index+1] = 1.0
		elif nn is 'G': values[length*index+2] = 1.0
		elif nn is 'T': values[length*index+3] = 1.0
		else: print('WARNING: Unrecognized codon '+nn+'.')
	return values

