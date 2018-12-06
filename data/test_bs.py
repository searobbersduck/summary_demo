# !/usr/bin/env python3

# from math import log
# from numpy import array
# from numpy import argmax
# import numpy as np
#
# # beam search
# def beam_search_decoder(data, k):
# 	sequences = [[list(), 1.0]]
# 	# walk over each step in sequence
# 	for row in data:
# 		all_candidates = list()
# 		# expand each current candidate
# 		for i in range(len(sequences)):
# 			seq, score = sequences[i]
# 			for j in range(len(row)):
# 				candidate = [seq + [j], score * -log(row[j])]
# 				all_candidates.append(candidate)
# 		# order all candidates by score
# 		ordered = sorted(all_candidates, key=lambda tup:tup[1])
# 		# select k best
# 		sequences = ordered[:k]
# 	return sequences
#
# # define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]
# data = array(data)
# # decode sequence
# result = beam_search_decoder(data, 3)
# # print result
# for seq in result:
# 	print(seq)
#
#
# # beam_width = k
# pred_out = []
# pred_in = [] # (k, T)
# pred = []


import numpy as np
import math

data = [[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1],
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0.5, 0.4, 0.3, 0.2, 0.1]]

data = np.array(data)

# input: (k, T)
# output: (k, T, c), c=5, T=10, k=3
def model(input):
	output = np.zeros((input.shape[0], input.shape[1], 5))
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			output[i][j][:] = data[np.random.randint(5)]
	return output

input = np.random.rand(5,10)

# output = model(input)


# input: preds
# input: all_candidates
# input: i_pos
# input: top k=3
# input: beam width w=4
def beam_search_decoder(preds, all_candidates_in, i_pos, k, w):
	all_candidates_out = list()
	for i in range(len(all_candidates_in)):
		seq, score = all_candidates_in[i]
		next_prob = preds[i][i_pos]
		dict_prob = {}
		for i,prob in enumerate(next_prob):
			dict_prob[i] = prob
		next_prob = sorted(dict_prob.items(), key=lambda x:x[1], reverse=True)[:w]
		for j in range(len(next_prob)):
			candidate = [seq + [next_prob[j][0]], score * -math.log(next_prob[j][1])]
			all_candidates_out.append(candidate)
	ordered = sorted(all_candidates_out, key=lambda tup: tup[1])
	return ordered[:k]

T = 20
k = 3
w = 4
all_candidates = [[list(), 1.0]]
inputs = np.random.rand(k, T)
for i in range(T):
	preds = model(inputs)
	all_candidates = beam_search_decoder(preds, all_candidates, i, k, w)
	for j, can in enumerate(all_candidates):
		seq, score = can
		inputs[j][i] = seq[i]




print('hello world!')
