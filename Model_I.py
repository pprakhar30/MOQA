import tensorflow as tf
import numpy as np
import os 
import psutil
import json
from random import randint
from math import ceil,log
from collections import defaultdict
from helper import sigmoid
import heapq

class Model:

	def __init__(self, k, numIter, corpus):

		self.PairwiseDim		= 3
		self.rankDim			= k
		self.V 					= corpus.Map.V
		self.Nq 				= len(corpus.QAnswers)
		self.Ns 				= len(corpus.Sentences)
		self.Na 				= 11
		self.Items 				= len(corpus.Map.ItemIDMap)
		self.numIter 			= numIter
		self.corpus 			= corpus
		self.valid 				= 0.8
		self.valid_test 		= 0.5
		self.nEpoch 			= 10
		self.valid_perf 		= -1.0
		self.theta 				= None
		self.RelvPar 			= None
		self.A 					= None
		self.B 					= None
		self.PredPar 			= None
		self.X 					= None
		self.Y 					= None
		self.optimizer 			= None
		self.correct 			= tf.Variable(0.0, dtype = tf.float64, trainable = False)
		self.best_valid_model 	= []
		self.trainQ 			= []
		self.validTestQ 		= []
		self.Pairwise 			= []
		self.Question 			= []
		self.Answer 			= []
		self.Review				= []
		self.TermtoTermR 		= []
		self.TermtoTermP 		= []
		self.ValidQuestion  	= []
		self.TestQuestions  	= []
		
		for i in range(self.Items):

			self.Pairwise.append([])
			self.Question.append([])
			self.Answer.append([])
			self.Review.append([])
			self.TermtoTermR.append([])
			self.TermtoTermP.append([])
			self.trainQ.append([])
			
		self.create_training_testing_data()

		
	def create_training_testing_data(self):

		for i in range(self.Items):

			X 		= self.corpus.QPerItem[i]
			for j in range(len(X)):
				if j <= int(self.valid* len(X)):
					self.trainQ[i].append(X[j])
				else:
					self.validTestQ.append(X[j])

		for i in range(self.Items):

			'Calculating Sparse Question Features'
			indices = []
			values 	= []
			X 		= self.trainQ[i]
			
			for j in range(int(len(X))):
				for k, count in sorted(self.corpus.QAnswers[X[j]].qFeature.items()):
					indices.append([j,k])
					values.append(count)
			
			shape 				= [len(X), self.V]
			self.Question[i] 	= tf.SparseTensor(np.array(indices), np.array(values), np.array(shape))

			
			'Calculating Sparse Answer and Sparse TermtoTermP features'
			indices1 	= []
			values1 	= []
			indices2 	= []
			values2 	= []
			X 			= self.trainQ[i]
			Y 			= self.corpus.SPerItem[i]
			
			for j in range(len(X)):
				for k in range(self.Na):
					if k==0:
						aFeature = self.corpus.QAnswers[X[j]].aFeature
					else:
						na = randint(0, len(X) - 1)
						if na == j:
							na = (na + 1) % len(X)
						aFeature = self.corpus.QAnswers[na].aFeature
					
					for l,count in sorted(aFeature.items()):
						indices1.append([j,k,l])
						values1.append(count)
						
					for m in range(len(Y)):
						sFeature = self.corpus.Sentences[Y[m]].sFeature
						for v1, c1 in sorted(aFeature.items()):
							if v1 in sFeature:
								indices2.append([j, k, m, v1])
								values2.append(c1 * sFeature[v1])
						
			
			shape1 				= [len(X), self.Na, self.V]
			shape2				= [len(X), self.Na, len(Y), self.V]
			self.Answer[i] 		= tf.SparseTensor(np.array(indices1), np.array(values1), np.array(shape1))
			self.TermtoTermP[i] = tf.SparseTensor(np.array(indices2), np.array(values2), np.array(shape2))
		
		
			'Calculating Sparse Review Features at Sentence Level'
			indices = []
			values 	= []
			X = self.corpus.SPerItem[i]
			
			for j in range(len(X)):
				for k, count in sorted(self.corpus.Sentences[X[j]].sFeature.items()):
					indices.append([j,k])
					values.append(count)
			
			shape 			= [len(X), self.V]
			self.Review[i] 	= tf.SparseTensor(indices, np.array(values), shape)

		
			'Calculating Dense PairWise and Sparse TermtoTermR features'
			X 				= self.trainQ[i]
			Y				= self.corpus.SPerItem[i]
			pairwise_temp 	= np.array(np.zeros((len(X), len(Y), self.PairwiseDim)), dtype = np.float64)
			indices 		= []
			values 			= []
			
			for j in range(len(X)):
				for k in range(len(Y)):
					pairwise_temp [j][k] 	= self.corpus.PairWiseFeature[(X[j], Y[k])]
					qFeature 				= self.corpus.QAnswers[X[j]].qFeature
					aFeature 				= self.corpus.Sentences[Y[k]].sFeature
					for v1, c1 in sorted(qFeature.items()):
						if v1 in aFeature:
							indices.append([j, k, v1])
							values.append(c1 * aFeature[v1])
							
			shape       			= [len(X), len(Y), self.V]
			self.Pairwise[i] 		= pairwise_temp
			self.TermtoTermR[i] 	= tf.SparseTensor(indices, np.array(values), shape)
	

	def calc_log_loss(self):

		log_likelihood = 0.0
		self.correct = 0
		
		for i in range(self.Items):
			
			print 'Doing for item %d'%(i)
			
			shape1 			= tf.shape(self.Pairwise[i])
			shape2 			= tf.shape(self.Answer[i])
			nq 				= shape1[0]
			nr 				= shape1[1]
			na 				= shape2[1]

			pairwise 		= tf.reshape(self.Pairwise[i], [-1, self.PairwiseDim])
			pairwise 		= tf.reshape(tf.matmul(pairwise, self.theta), [nq, nr])
	
			termTotermR 	= tf.sparse_reshape(self.TermtoTermR[i], [-1, self.V])
			termTotermR 	= tf.reshape(tf.sparse_tensor_dense_matmul(termTotermR, self.RelvPar), [nq, nr])
	
			QProj			= tf.sparse_tensor_dense_matmul(self.Question[i], self.A)
			RProjR			= tf.sparse_tensor_dense_matmul(self.Review[i], self.B)
			BilinearR		= tf.matmul(QProj, tf.transpose(RProjR))
	
			Relevance		= tf.nn.softmax(pairwise + termTotermR + BilinearR)
	
			termTotermP 	= tf.sparse_reshape(self.TermtoTermP[i], [-1, self.V])
			termTotermP 	= tf.reshape(tf.sparse_tensor_dense_matmul(termTotermP, self.PredPar), [nq, na, nr])
	
			AProj			= tf.sparse_tensor_dense_matmul(tf.sparse_reshape(self.Answer[i], [-1, self.V]), self.X)
			RProjP			= tf.sparse_tensor_dense_matmul(self.	Review[i], self.Y)
			BilinearP		= tf.reshape(tf.matmul(AProj, tf.transpose(RProjP)), [nq, na, nr])
			
			Prediction 		= BilinearP + termTotermP
			Prediction  	= tf.expand_dims(Prediction[:,0,:], 1) - Prediction
			Prediction		= Prediction[:,1:,:]
			Prediction		= tf.sigmoid(Prediction)
			
			MoE 			= tf.multiply(Prediction, tf.expand_dims(Relevance, axis = 1))
			MoE 			= tf.log(tf.reduce_sum(MoE, axis = 2))

			self.correct 	= tf.shape(tf.where(MoE, 0.5))[0]

			log_likelihood  += tf.reduce_sum(MoE)

		
		return -1*log_likelihood

	
	def unflatten(self, param):

		var_vals 	= [param[packing_slice] for packing_slice in self.optimizer._packing_slices]
		theta 		= np.reshape(var_vals[0], (self.PairwiseDim, 1))
		RelvPar 	= np.reshape(var_vals[1], (self.V, 1))
		A 			= np.reshape(var_vals[2], (self.V, self.rankDim))
		B 			= np.reshape(var_vals[3], (self.V, self.rankDim))
		PredPar 	= np.reshape(var_vals[4], (self.V, 1))
		X 			= np.reshape(var_vals[5], (self.V, self.rankDim))
		Y 			= np.reshape(var_vals[6], (self.V, self.rankDim))

		return theta, RelvPar, A, B, PredPar, X, Y

	
	def calc_relevance_score(self, question, sent, params):
			
		pairWiseSim 	= np.matmul(self.corpus.PairWiseFeature[(question,sent)], params[0])
		Question    	= self.corpus.QAnswers[question].qFeature
		Sent 	    	= self.corpus.Sentences[sent].sFeature
		termToterm  	= np.matmul(np.multiply(Question, Sent), params[1])
		bilinear    	= np.matmul(np.matmul(Question, params[2]), np.transpose(np.matmul(Sent, params[3])))
		rel_score 		= (pairWiseSim + termToterm + bilinear)[0][0]
		
		return rel_score

	
	def calc_pred_score(self, answer, nanswer,  sentence, params):

		Answer 		= self.corpus.QAnswers[answer].aFeature
		NAnswer 	= self.corpus.QAnswers[nanswer].aFeature
		Sentence 	= self.corpus.Sentences[sentence].sFeature
		
		termToterm1	= np.matmul(np.multiply(Answer, Sentence), params[4])
		bilinear1	= np.matmul(np.matmul(Answer, params[5]), np.transpose(np.matmul(Sentence, params[6])))
		pred_score1	= termToterm1 + bilinear1

		termToterm2	= np.matmul(np.multiply(NAnswer, Sentence), params[4])
		bilinear2	= np.matmul(np.matmul(NAnswer, params[5]), np.transpose(np.matmul(Sentence, params[6])))
		pred_score2	= termToterm2 + bilinear2

		pred_score  = pred_score1 - pred_score2
		pred_score  = pred_score[0][0]

		return sigmoid(pred_score)

	def AUC(self):

		params 		= [self.theta, self.RelvPar, self.A, self.B, self.PredPar, self.X, self.Y]
		nq 			= len(self.validTestQ)
		AUC 		= [0] * nq
		AUC_valid 	= 0
		AUC_test 	= 0
		test 		= int(nq * self.valid_test)
		max_na 		= 1000
		
		for q in range(nq):

			if q < test:
				na_start = 0
				na_end = test
			else:
				na_start = test
				na_end = nq

			if na_end - na_start + 1 > max_na:
				na_end = na_start + max_na

			ques 				= self.validTestQ[q]
			itemId 				= self.corpus.QAnswers[ques].itemId
			prediction 			= [0] * (na_end-na_start)
			
			for s in range(len(self.corpus.SPerItem[itemId])):

				sent 		= self.corpus.SPerItem[itemId][s]
				rel_score 	= self.calc_relevance_score(ques, sent, params)
				rel_score 	= np.exp(rel_score)
				
				if rel_score == np.inf:
					rel_score 	= np.exp(200)

				Z += rel_score

				for na in range(na_start, na_end):

					if na == q:
						continue
					prediction[na - na_start] += rel_score * self.calc_pred_score(ques, self.validTestQ[na], sent, params)

			for na in range(na_start, na_end):

				if prediction[na- na_start] / Z > 0.5:
					AUC[q] += 1

			AUC[q] /= (na_end - na_start)

		for q in range(nq):

			if q < test:
				AUC_valid += AUC[q]

			else:
				AUC_test += AUC[q]

		AUC_valid 	/= test 
		AUC_test 	/= (nq - test)

		return AUC_valid, AUC_test

				
	def valid_test_perf(self, param):

		test 			= int(len(self.validTestQ) * self.valid_test) 
		params 			= self.unflatten(param)
		MostRelevant 	= [None] * len(self.validTestQ)
		CorrectV 		= 0
		CorrectT 		= 0

		for q in range(len(self.validTestQ)):

			ques 				= self.validTestQ[q]
			itemId 				= self.corpus.QAnswers[ques].itemId
			most_relv_score 	= -1
			pred_score 			= [None] * len(self.corpus.SPerItem[itemId])
			prediction 			= 0.0
			Z 					= 0.0
			
			if q < test:
				na = randint(0, test - 1)
				if na == q:
					na = (na + 1) % test

			else : 
				na = randint(test, len(self.validTestQ) - 1)
				if na == q:
					if na == len(self.validTestQ) - 1:
						na = test
					else: 
						na = na + 1

			na 	= self.validTestQ[na]

			for s in range(len(self.corpus.SPerItem[itemId])):

				sent 		= self.corpus.SPerItem[itemId][s]
				rel_score 	= self.calc_relevance_score(ques, sent, params)
				
				if MostRelevant[q] is None:
					
					MostRelevant[q] = sent
					most_relv_score = rel_score 
				
				elif  most_relv_score < rel_score:
					
					MostRelevant[q] = sent
					most_relv_score = rel_score
				
				rel_score 	= np.exp(rel_score)
				
				if rel_score == np.inf:
					rel_score 	= np.exp(200)

				Z 				+= rel_score
				pred_score[s] 	 = rel_score * self.calc_pred_score(ques, na, sent, params)

			for s in range(len(self.corpus.SPerItem[itemId])):
				pred_score[s] = pred_score[s] / Z
				prediction += pred_score

			if prediction > 0.5:
				if q < test:
					CorrectV += 1
				else:
					 CorrectT += 1

		valid_accuracy 	= (CorrectV * 1.0) / test
		test_accuracy 	= (CorrectT * 1.0) / (len(self.validTestQ) - test)
		
		return valid_accuracy, test_accuracy, MostRelevant

	
	def callback(self, param):

		train_Q 									= (self.Nq - len(self.validTestQ)) * 1.0
		train_accuracy 								= self.correct.eval() / trainQ
		valid_accuracy, test_accuracy, MostRelevant = self.valid_test_perf(param)

		if self.valid_perf <= valid_accuracy:

			self.valid_perf 		= valid_accuracy
			self.best_valid_model 	= param

		print '--------------------------------------------\n'
		print 'Training Accuracy: '+str(train_accuracy)
		print 'Validation Accuracy: '+str(valid_accuracy)
		print 'Test Accuracy: '+str(test_accuracy)
		print '---------------------------------------------'

	def top_ranked(self, Ktop):

		topRanked = [None] * len(self.validTestQ)
		for ques in self.validTestQ:

			h = []
			itemId = self.corpus.QAnswers[ques].itemId
			for sent in self.corpus.SPerItem[itemId]:

				relv_score = self.calc_relevance_score(ques, sent)
				if len(h) == Ktop:
					heapq.heapreplace(h, (relv_score, sent))
				else:
					heapq.heappush((relv_score,sent))
			topRanked.append(sorted(h))

		return topRanked


	def load_model(self):

		theta, RelvPar, A, B, PredPar, X, Y = self.unflatten(self.best_valid_model)
		
		self.theta 		= theta
		self.RelvPar 	= RelvPar
		self.A 			= A
		self.B 			= B
		self.PredPar 	= PredPar
		self.X 			= X
		self.Y 			= Y

	
	def save_predictions(self, MostRelevant, file):

		with open(file, 'w') as file:
			
			maxi = 1000
			for q in range(len(self.validTestQ)-1, 0, -1):

				ques = self.validTestQ[q]
				sent = MostRelevant[q]
				json_dump = {
								 'itemId': str(self.Map.RItemIDMap[self.corpus.QAnswers[ques].itemId]),
								 'Question': self.corpus.QAnswers[ques].question,
								 'Answer': self.corpus.QAnswers[ques].answer,
								 'Review': self.corpus.Sentences[sent].rObj.reviewText
								 'Sanitized': self.corpus.Sentences[sent].sent 
					 		}

				json.dump(json_dump, file, indent = 4)

	
	def save_top_ranked(self, topRanked, file):

		with open(file, 'w') as file:

			for q in range(len(self.validTestQ)-1):
				
				ques 		= self.validTestQ[q]
				json_dump 	= {
								 'itemId': str(self.Map.RItemIDMap[self.corpus.QAnswers[ques].itemId]),
								 'Question': self.corpus.QAnswers[ques].question,
							 	 'Answer': self.corpus.QAnswers[ques].answer,
							  }
				for j in range(len(topRanked[q])):

					score, sent = topRanked[q][j]
					sub_dump = {
									'Relevance': score,
									'Sentence': self.corpus.Sentences[sent].sent,
									'Review': self.corpus.Sentences[sent].rObj.reviewText 
								}
					
					json_dump['Sent'+str(j)] = sub_dump
				json.dump(json_dump, file, indent = 4)


	def save_model(self, file):

		saver = tf.train.Saver()
		sess = tf.get_default_session()
		saver.save(sess, file)

	
	def restore_model(self, file):

		sess = tf.Session()
		metafile = file+'.meta'
		new_saver = tf.train.import_meta_graph(metafile)
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		all_vars = tf.get_collection('vars')
		for v in all_vars:
		    v_ = sess.run(v)
		    print(v_)



	def train_model(self):

			self.theta   	= tf.Variable(tf.random_uniform([self.PairwiseDim, 1], dtype = tf.float64), name = 'theta' )
			self.RelvPar 	= tf.Variable(tf.random_uniform([self.V, 1], dtype = tf.float64), name = 'RelvPar')
			self.A       	= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'A')
			self.B		 	= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'B')
			self.PredPar 	= tf.Variable(tf.random_uniform([self.V, 1], dtype = tf.float64), name = 'PredPar')
			self.X       	= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'X')
			self.Y		 	= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'Y')

			loss 			= self.calc_log_loss()
			self.optimizer  = tf.contrib.opt.ScipyOptimizerInterface(loss,
	                           method='L-BFGS-B', options={'disp': True})
			
			with tf.Session() as sess:

				sess.run(tf.global_variables_initializer())
				self.optimizer.minimize(sess, step_callback = self.callback)
				

			

			 
			


		
