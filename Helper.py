import autograd.numpy as np
from collections import namedtuple,defaultdict


def disp(Disp):

	print Disp+" .........."
	print "."
	print "."
	print "."
	print "."


def calc_LCS(str1, str2):

	m = len(str1)+1
	n = len(str2)+1
	LCS = np.zeros((m,n), dtype= np.int32)
	for i in range(1,m):
		for j in range(1,n):
			if str1[i-1] == str2[j-1]:
				LCS[i][j] = LCS[i-1][j-1] + 1
			else:
				LCS[i][j] = max(LCS[i-1][j],LCS[i][j-1])
	return LCS[m-1][n-1]


def sigmoid(x):

	return 1.0/(1.0 + np.exp(x))

def check_sent(sent, WordIDMap):

	x = [WordIDMap[word] for word in sent.split(' ') if word in WordIDMap]
	if len(x) > 0:
		return True
	else:
		return False

def normalize(feature):
	
	L = feature.values()
	L = sum([x*x for x in L])
	for k, v  in feature.items():
		feature[k] = v/L

def split_sparse(p ,tr, q, a, tp, Nq, start, end):

	sp_p 	= tf.sparse_split(sp_input = p, num_split=Nq, axis=0)
	sp_tr 	= tf.sparse_split(sp_input = tr, num_split=Nq, axis=0)
	sp_q 	= tf.sparse_split(sp_input = q, num_split=Nq, axis=0)
	sp_a 	= tf.sparse_split(sp_input = a, num_split=Nq, axis=0)
	sp_tp 	= tf.sparse_split(sp_input = tp, num_split=Nq, axis=0)

	cs_p 	= tf.sparse_concat(axis=0, sp_inputs=sp_p[start:end])
	cs_tr 	= tf.sparse_concat(axis=0, sp_inputs=sp_tr[start:end])
	cs_q 	= tf.sparse_concat(axis=0, sp_inputs=st_q[start:end])
	cs_a 	= tf.sparse_concat(axis=0, sp_inputs=st_a[start:end])
	cs_tp 	= tf.sparse_concat(axis=0, sp_inputs=st_tp[start:end])

	return cs_p, cs_tr, cs_q, cs_a, cs_tp





