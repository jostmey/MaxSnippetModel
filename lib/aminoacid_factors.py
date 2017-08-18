##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2016-05-02
# Environment: Python3
# License: See LICENSE
# Purpose: Tools for representing an amino acid sequence as a sequence of categorical variables.
##########################################################################################

labels = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
length = 20

def features(sequence):
	values = [0.0]*(length*len(sequence))
	for index, aa in enumerate(sequence):
		if aa is 'X':
			continue
		offset = labels.index(aa)
		values[length*index+offset] = 1.0
	return values

