import numpy as np

def calc_LCS(str1, str2):

	m 	= len(str1)+1
	n 	= len(str2)+1
	LCS = np.zeros((m,n), dtype= np.int32)
	
	for i in range(1,m):
		for j in range(1,n):
			
			if str1[i-1] == str2[j-1]:
				LCS[i][j] = LCS[i-1][j-1] + 1
			
			else:
				LCS[i][j] = max(LCS[i-1][j],LCS[i][j-1])
	
	return LCS[m-1][n-1]


def check_sent(sent, WordIDMap):

	x = [WordIDMap[word] for word in sent.split(' ') if word in WordIDMap]
	
	if len(x) > 0:
		return True
	else:
		return False


def normalize(feature):
	
	L 	= feature.values()
	norm 	= np.sqrt(np.sum(np.power(L, 2)))
				
	for k, v  in feature.items():
		feature[k] = v/norm
