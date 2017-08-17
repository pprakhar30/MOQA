import json
import numpy as np
import tensorflow as tf

from math import ceil,log
from random import randint
from collections import defaultdict, OrderedDict

class Model:

	def __init__(self, k, numIter, Lambda ,corpus):

		'''
		Parameters
		-------------

		k: the rank of the matrices which approximate the matrix M in qMd'. Such that M = AB', where each A and B have rank 'k'
		numIter: Number of iterations you want to train your model
		Lambda: regularization parameter
		corpus: an object of Corpus class

		Model Parameters
		-----------------
		theta: parameter corresponding to pairwise similarity
		RelvPar: parameter corresponding to term to term Relevance of question and review
		A: parameter corresponding to the bilinear term, projects question to 'k' dimensional space
		B, Y: parameter corresponding to the bilinear term, projects review to 'k' dimensional space
		PredPar: parameter corresponding to term to term Relevance of answer and review
		X:  parameter corresponding to the bilinear term, projects answer to 'k' dimensional space
		
		Other Attributes:
		-----------------
		PairwiseDim: takes bm25, bm25+ and LCS as three off the plate similarity measures and hence 3
		V: voublary size of the corpus we have read 
		Na: Each answer matrix contains 11 answer 0th being the right answer and other 10 being the wrong answer for training purpose. Total answers hence is 11 by default.
		Items: Our training is done item wise each item define its own set of documents 
		valid: the training validation split
		valid_test: the test training split
		trainQPerItem: Stores the Ids of the question used for training
		validTestQ: Stores the Ids of the question in validation and test set
		validTestNa: Stores the ids of the non answer to be used to evalueate the validation and test set performance during training.
		validTestM: Stores the Sparse Matrix representing all the questions, answer, review, term to term similarity and pairwise similarity for each question in validation and test set
		Pairwise: Stores the pairwise feature for each item.
		Question: Stores the Sparse Matrix representing all the questions in a item for each item
		Answer: Stores the Sparse Matrix representing all the answers (also the 10 non-answer) corresponding to a question in a item for each item
		Review: Stores the Sparse Matrix representing all the reviews in a item for each item
		TermtoTermR: Stores the Sparse Matrix representing all the term to term similarity between question and reviews in a item for each item
		TermtoTermP: Stores the Sparse Matrix representing all the term to term similarity between answer and reviews in a item for each item
		Question_I: Sparse Bit mask for Question Matrix above
		Answer_I: Sparse Bit mask for Answer Matrix above
		Review_I: Sparse Bit mask for Review Matrix above
		pairwise, question, answer, review, question_I, answer_I, review_I, termTotermR, termTotermP: Placeholder for above matrices			
		
		'''

		self.PairwiseDim		= 3
		self.rankDim			= k
		self.V 					= corpus.Map.V
		self.Na 				= 11
		self.Items 				= len(corpus.Map.ItemIDMap)
		self.numIter 			= numIter
		self.Lambda 			= Lambda
		self.corpus 			= corpus
		self.valid 				= 0.8
		self.valid_test 		= 0.5
		self.theta 				= None
		self.RelvPar 			= None
		self.A 					= None
		self.B 					= None
		self.PredPar 			= None
		self.X 					= None
		self.Y 					= None
		self.loss 				= None
		self.trainQPerItem 		= []
		self.trainQ 			= []
		self.validTestQ 		= []
		self.validTestNa		= []
		self.validTestM			= []
		self.Pairwise 			= []
		self.Question 			= []
		self.Answer 			= []
		self.Review				= []
		self.TermtoTermR 		= []
		self.TermtoTermP 		= []
		self.Question_I			= []
		self.Answer_I 			= []
		self.Review_I			= []
		self.pairwise			= tf.placeholder(dtype = tf.float64, name = 'Pairwise')
		self.question 			= tf.sparse_placeholder(dtype = tf.float64, name = 'Question')
		self.answer 			= tf.sparse_placeholder(dtype = tf.float64, name = 'Answer')
		self.review 			= tf.sparse_placeholder(dtype = tf.float64, name = 'Review')
		self.question_I			= tf.sparse_placeholder(dtype = tf.float64, name = 'Question')
		self.answer_I 			= tf.sparse_placeholder(dtype = tf.float64, name = 'Answer')
		self.review_I 			= tf.sparse_placeholder(dtype = tf.float64, name = 'Review')
		self.termTotermR		= tf.sparse_placeholder(dtype = tf.float64, name = 'Review')
		self.termTotermP		= tf.sparse_placeholder(dtype = tf.float64, name = 'Review')
						
		self.initialize()
		self.create_training_data()
		self.create_validTest_data()
		
	
	def initialize(self):

		for i in range(self.Items):
			self.Pairwise.append([])
			self.Question.append([])
			self.Answer.append([])
			self.Review.append([])
			self.TermtoTermR.append([])
			self.TermtoTermP.append([])
			self.trainQPerItem.append([])
			self.Question_I.append([])
			self.Answer_I.append([])
			self.Review_I.append([])
		
	def create_sparse_one(self, qFeature = None, answer_list = None):

		indices = []
		values 	= []
		
		if answer_list is None:
			for k, count in sorted(qFeature.items()):
				indices.append([0, k])
				values.append(count)

			if indices == []:
				indices.append([0,0])
				values.append(np.float64(0))

			shape = [1, self.V]
				
			return (np.array(indices), np.array(values), np.array(shape))

		else:
			for i in range(len(answer_list)):
				aFeature = self.corpus.QAnswers[answer_list[i]].aFeature
				
				for k, count in sorted(aFeature.items()):
					indices.append([0, i, k])
					values.append(count)

			if indices == []:
				indices.append([0,0,0])
				values.append(np.float64(0))

			shape = [1, len(answer_list), self.V]

			return (np.array(indices), np.array(values), np.array(shape))

	def create_sparse_two(self, item, qFeature = None, answer_list = None):

		indices = []
		values  = []
		Y 		= self.corpus.SPerItem[item]

		if answer_list is None:
			for i in range(len(Y)):
				sFeature = self.corpus.Sentences[Y[i]].sFeature
				
				for v1, c1 in sorted(qFeature.items()):
							if v1 in sFeature:
								indices.append([0, i, v1])
								values.append(c1 * sFeature[v1])

			if indices == []:
				indices.append([0,0,0])
				values.append(np.float64(0))

			shape = [1, len(Y), self.V]
			
			return (np.array(indices), np.array(values), np.array(shape))

		else:
			for i in range(len(answer_list)):
				aFeature = self.corpus.QAnswers[answer_list[i]].aFeature
				
				for j in range(len(Y)):
					sFeature = self.corpus.Sentences[Y[j]].sFeature
					
					for v1, c1 in sorted(aFeature.items()):
						if v1 in sFeature:
							indices.append([0, i, j, v1])
							values.append(c1 * sFeature[v1])
			
			if indices == []:
				indices.append([0,0,0,0])
				values.append(np.float64(0))

			shape = [1, len(answer_list), len(Y), self.V]
			
			return (np.array(indices), np.array(values), np.array(shape))

	
	def create_dense_pairwise(self, item, qId):

		Y 			= self.corpus.SPerItem[item]
		Pairwise 	= np.zeros((1, len(Y), self.PairwiseDim), dtype = np.float64)
		
		for j in range(len(Y)):
			Pairwise[0][j] = self.corpus.PairWiseFeature[(qId, Y[j])]
		
		return Pairwise

	
	def create_validTest_data(self):

		for i in range(len(self.validTestQ)):
			qId 		= self.validTestQ[i]
			item 		= self.corpus.QAnswers[qId].itemId
			question 	= self.corpus.QAnswers[qId].qFeature
			answer_list	= [qId, self.validTestNa[i]]
			
			Pairwise 	= self.create_dense_pairwise(item, qId)
			Question 	= self.create_sparse_one(qFeature = question)
			Answer 		= self.create_sparse_one(answer_list = answer_list) 
			Review 		= self.Review[item]
			TermtoTermR = self.create_sparse_two(item, qFeature = question)
			TermtoTermP = self.create_sparse_two(item, answer_list = answer_list)

			Question_I  = (Question[0], Question[1] if Question[1].size == 1 and Question[1][0] == 0 else np.full((Question[1].size), 1.0/np.sqrt(Question[1].size)), Question[2])
			Answer_I    = (Answer[0], Answer[1] if Answer[1].size == 1 and Answer[1][0] == 0 else np.full((Answer[1].size), 1.0/np.sqrt(Answer[1].size)), Answer[2])
			Review_I 	= (Review[0], np.full((Review[1].size), 1.0/np.sqrt(Review[1].size)), Review[2])
			
			self.validTestM.append((Pairwise, Question, Answer, Review, TermtoTermR, TermtoTermP, Question_I, Answer_I, Review_I))

	
	def create_training_data(self):

		for i in range(self.Items):
			X 		= self.corpus.QPerItem[i]
			
			for j in range(len(X)):
				if j < int(ceil(self.valid* len(X))):
					self.trainQPerItem[i].append(X[j])
					self.trainQ.append(X[j])
				
				else:
					self.validTestQ.append(X[j])

		test = int(ceil(len(self.validTestQ) * self.valid_test))

		for i in range(len(self.validTestQ)):
			if i < test:
				na = randint(0, test - 1)
				
				if na == i:
					na = (na + 1) % test

			else : 
				na = randint(test, len(self.validTestQ) - 1)
				if na == i:
					if na == len(self.validTestQ) - 1:
						na = test
					
					else: 
						na = na + 1
			
			self.validTestNa.append(self.validTestQ[na]) 
		
		for i in range(self.Items):

			print "Creating data for ",i

			'Calculating Sparse Question Features'
			indices = []
			values 	= []
			X 		= self.trainQPerItem[i]
			
			for j in range(int(len(X))):
				for k, count in sorted(self.corpus.QAnswers[X[j]].qFeature.items()):
					indices.append([j,k])
					values.append(count)

			shape 				= [len(X), self.V]
			self.Question[i] 	= (np.array(indices), np.array(values), np.array(shape))
			self.Question_I[i]  = (np.array(indices), np.full((len(values)), 1.0/np.sqrt(len(values))), np.array(shape))
						
			'Calculating Sparse Answer and Sparse TermtoTermP features'
			indices1 	= []
			values1 	= []
			indices2 	= []
			values2 	= []
			X 			= self.trainQPerItem[i]
			Y 			= self.corpus.SPerItem[i]
			
			for j in range(len(X)):
				for k in range(self.Na):
					if k==0:
						aFeature = self.corpus.QAnswers[X[j]].aFeature
					
					else:
						na = randint(0, len(self.trainQ) - 1)
						
						if self.trainQ[na] == X[j]:
							na = (na + 1) % len(self.trainQ)
						
						aFeature = self.corpus.QAnswers[self.trainQ[na]].aFeature
					
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
			self.Answer[i] 		= (np.array(indices1), np.array(values1), np.array(shape1))
			self.Answer_I[i]    = (np.array(indices1), np.full((len(values1)), 1.0/np.sqrt(len(values1))), np.array(shape1))
			self.TermtoTermP[i] = (np.array(indices2), np.array(values2), np.array(shape2))
		
		
			'Calculating Sparse Review Features at Sentence Level'
			indices = []
			values 	= []
			X 		= self.corpus.SPerItem[i]
			
			for j in range(len(X)):
				for k, count in sorted(self.corpus.Sentences[X[j]].sFeature.items()):
					indices.append([j,k])
					values.append(count)
			
			shape 				= [len(X), self.V]
			self.Review[i] 		= (np.array(indices), np.array(values), np.array(shape))
			self.Review_I[i]	= (np.array(indices), np.full((len(values)), 1.0/np.sqrt(len(values))), np.array(shape))

		
			'Calculating Dense PairWise and Sparse TermtoTermR features'
			X 				= self.trainQPerItem[i]
			Y				= self.corpus.SPerItem[i]
			pairwise_temp 	= np.zeros((len(X), len(Y), self.PairwiseDim), dtype = np.float64)
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
			self.TermtoTermR[i] 	= (np.array(indices), np.array(values), np.array(shape))

			
		
	def calc_log_loss(self, Pairwise, Question, Answer, Review, TermtoTermR, TermtoTermP, Question_I, Answer_I, Review_I):

		#print 'Doing for item %d'%(i)
		
		shape1 			= tf.shape(Pairwise)
		shape2 			= tf.shape(Answer)

		nq 				= shape1[0]
		nr 				= shape1[1]
		na 				= shape2[1]

		pairwise 		= tf.reshape(Pairwise, [-1, self.PairwiseDim])
		pairwise 		= tf.reshape(tf.matmul(pairwise, self.theta), [nq, nr])

		termTotermR 	= tf.sparse_reshape(TermtoTermR, [-1, self.V])
		termTotermR 	= tf.reshape(tf.sparse_tensor_dense_matmul(termTotermR, self.RelvPar), [nq, nr])

		QProj			= tf.sparse_tensor_dense_matmul(Question_I, self.A)
		RProjR			= tf.sparse_tensor_dense_matmul(Review_I, self.B)
		BilinearR		= tf.matmul(QProj, tf.transpose(RProjR))

		Relevance		= tf.nn.softmax(pairwise + termTotermR + BilinearR)

		termTotermP 	= tf.sparse_reshape(TermtoTermP, [-1, self.V])
		termTotermP 	= tf.reshape(tf.sparse_tensor_dense_matmul(termTotermP, self.PredPar), [nq, na, nr])

		AProj			= tf.sparse_tensor_dense_matmul(tf.sparse_reshape(Answer_I, [-1, self.V]), self.X)
		RProjP			= tf.sparse_tensor_dense_matmul(Review_I, self.Y)
		BilinearP		= tf.reshape(tf.matmul(AProj, tf.transpose(RProjP)), [nq, na, nr])
		
		Prediction 		= BilinearP + termTotermP
		Prediction  	= tf.expand_dims(Prediction[:,0,:], 1) - Prediction
		Prediction		= Prediction[:,1:,:]
		Prediction		= tf.sigmoid(Prediction)
		
		MoE 			= tf.reduce_sum(tf.multiply(Prediction, tf.expand_dims(Relevance, axis = 1)), axis = 2)
		accuracy_count  = tf.cast(tf.shape(tf.where(MoE > 0.5))[0], tf.float64)
		count 			= nq * na 
		
		log_likelihood  = tf.reduce_sum(tf.log(MoE))
		R1 				= tf.reduce_sum(tf.square(self.A)) + tf.reduce_sum(tf.square(self.B)) 
		R2				= tf.reduce_sum(tf.square(self.X)) + tf.reduce_sum(tf.square(self.Y))
		
		log_likelihood  -= self.Lambda * (R1 + R2)

		return -1*log_likelihood, MoE, Relevance

	def AUC(self, sess):

		nq 			= len(self.validTestQ)
		AUC 		= [0] * nq
		AUC_valid 	= 0
		AUC_test 	= 0
		test 		= int(ceil(len(self.validTestQ) * self.valid_test))
		max_na 		= 1000
		
		for q in range(nq):
			print "Calculating AUC for ",q
			
			if q < test:
				na_start 	= 0
				na_end 		= test
			
			else:
				na_start 	= test
				na_end 		= nq

			if (na_end - na_start) > max_na:
				na_end = na_start + max_na

			pairwise, question, answer, review, termtoTermR, termtoTermP, question_I, answer_I, review_I 	= self.validTestM[q]

			itemId 			= self.corpus.QAnswers[self.validTestQ[q]].itemId
			answer_list 	= self.validTestQ[na_start:na_end]
			
			if self.validTestQ[q] in answer_list:
				answer_list.remove(self.validTestQ[q])
			
			answer_list 	= [self.validTestQ[q]] + answer_list
			answer 			= self.create_sparse_one(answer_list = answer_list)
			answer_I 		= (answer[0], np.full((answer[1].size), 1.0/np.sqrt(answer[1].size)), answer[2])
			termtoTermP 	= self.create_sparse_two(itemId, answer_list = answer_list)
			
			feed_dict = {
						 self.pairwise 		: pairwise,
						 self.question 		: question,
						 self.answer 		: answer,
						 self.review   		: review,
						 self.termTotermR 	: termtoTermR,
						 self.termTotermP 	: termtoTermP,
						 self.question_I 	: question_I,
						 self.answer_I 		: answer_I,
						 self.review_I 		: review_I
						 }
			
			log_likelihood, MoE, Relevance 	= sess.run(self.loss, feed_dict = feed_dict)
			correct 						= len(MoE[np.where(MoE > 0.5)])
			accuracy 						= (correct * 1.0) / (len(answer_list) - 1)

			if q < test:
				AUC_valid += accuracy
			
			else:
				AUC_test += accuracy

		AUC_valid /= test 
		AUC_test /= (nq - test)	

		return AUC_valid, AUC_test

				
	def valid_test_perf(self, sess = None):

		test 			= int(ceil(len(self.validTestQ) * self.valid_test))
		MostRelevant 	= [None] * len(self.validTestQ)
		CorrectV 		= 0
		CorrectT 		= 0

		for i in range(len(self.validTestM)):
			pairwise, question, answer, review, termtoTermR, termtoTermP, question_I, answer_I, review_I 	= self.validTestM[i]
			
			feed_dict = {
						 self.pairwise 		: pairwise,
						 self.question 		: question,
						 self.answer 		: answer,
						 self.review   		: review,
						 self.termTotermR 	: termtoTermR,
						 self.termTotermP 	: termtoTermP,
						 self.question_I 	: question_I,
						 self.answer_I 		: answer_I,
						 self.review_I 		: review_I
						 }
			
			log_likelihood, MoE, Relevance 	= sess.run(self.loss, feed_dict = feed_dict)
			
			if i < test:
				CorrectV += len(MoE[np.where(MoE > 0.5)])

			else:
				CorrectT += len(MoE[np.where(MoE > 0.5)])

			ind 			= np.argmax(Relevance)
			item 			= self.corpus.QAnswers[self.validTestQ[i]].itemId
			sent 			= self.corpus.SPerItem[item][ind]
			MostRelevant[i] = sent 

		valid_accuracy 	= (CorrectV * 1.0) / test
		test_accuracy	= (CorrectT * 1.0) / (len(self.validTestQ) - test)

		return valid_accuracy, test_accuracy, MostRelevant

	
	def top_ranked(self, sess, Ktop = 10):

		topRanked 	= [] 
		
		for i in range(len(self.validTestM)):
			h 																= []
			itemId 															= self.corpus.QAnswers[self.validTestQ[i]].itemId
			
			pairwise, question, answer, review, termtoTermR, termtoTermP, question_I, answer_I, review_I 	= self.validTestM[i]
			
			feed_dict = {
						 self.pairwise 		: pairwise,
						 self.question 		: question,
						 self.answer 		: answer,
						 self.review   		: review,
						 self.termTotermR 	: termtoTermR,
						 self.termTotermP 	: termtoTermP,
						 self.question_I 	: question_I,
						 self.answer_I 		: answer_I,
						 self.review_I 		: review_I
						 }

			log_likelihood, MoE, Relevance 	= sess.run(self.loss, feed_dict = feed_dict)
			Relevance 						= Relevance[0]
			Relevance 						= sorted([(Relevance[i], i) for i in range(len(Relevance))])
			
			if len(Relevance) > Ktop:
				Relevance = Relevance[len(Relevance)-Ktop:]

			for score,ind in Relevance:
				sent = self.corpus.SPerItem[itemId][ind]
				h.append((score,sent))

			topRanked.append(h)

		return topRanked

	def save_predictions(self, MostRelevant, file):

		with open(file, 'w') as file:
			maxi = 1000
			
			for q in range(len(self.validTestQ)-1, -1, -1):
				ques = self.validTestQ[q]
				sent = MostRelevant[q]
				
				json_dump = OrderedDict(
										 [('itemId', self.corpus.Map.RItemIDMap[self.corpus.QAnswers[ques].itemId]),
										 ('Question', self.corpus.QAnswers[ques].question),
										 ('Answer', self.corpus.QAnswers[ques].answer),
										 ('Review', self.corpus.Sentences[sent].rObj.reviewText),
										 ('Sanitized', self.corpus.Sentences[sent].sent)] 
					 					)

				json.dump(json_dump, file, indent = 4)

	
	def save_top_ranked(self, topRanked, file):

		with open(file, 'w') as file:
			for q in range(len(self.validTestQ)):
				ques 		= self.validTestQ[q]
				json_dump 	= OrderedDict(
										 [('itemId', self.corpus.Map.RItemIDMap[self.corpus.QAnswers[ques].itemId]),
										  ('Question', self.corpus.QAnswers[ques].question),
									 	  ('Answer', self.corpus.QAnswers[ques].answer)]
							  			 )
				
				for j in range(len(topRanked[q])-1 , -1, -1):
					score, sent = topRanked[q][j]
					sub_dump = OrderedDict(
											[('Relevance', score),
											 ('Sentence', self.corpus.Sentences[sent].sent),
											 ('Review', self.corpus.Sentences[sent].rObj.reviewText)] 
										   )
					
					json_dump['Sent'+str(j)] = sub_dump
				
				json.dump(json_dump, file, indent = 4)


	def save_model(self, file):

		saver 	= tf.train.Saver()
		sess 	= tf.get_default_session()
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

			self.theta   		= tf.Variable(tf.random_uniform([self.PairwiseDim, 1], dtype = tf.float64), name = 'theta' )
			self.RelvPar 		= tf.Variable(tf.random_uniform([self.V, 1], dtype = tf.float64), name = 'RelvPar')
			self.A       		= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'A')
			self.B		 		= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'B')
			self.PredPar 		= tf.Variable(tf.random_uniform([self.V, 1], dtype = tf.float64), name = 'PredPar')
			self.X       		= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'X')
			self.Y		 		= tf.Variable(tf.random_uniform([self.V, self.rankDim], dtype = tf.float64), name = 'Y')
			
			self.loss		= self.calc_log_loss(self.pairwise, self.question, self.answer, self.review, self.termTotermR, self.termTotermP, self.question_I, self.answer_I, self.review_I)
			train_step 		= tf.train.AdamOptimizer(1e-2).minimize(self.loss[0])
    		
			sess = tf.Session()

			sess.run(tf.global_variables_initializer())
			for i in range(self.numIter):
				log_likelihood 	= 0.0
				accuracy_count	= 0
				count 			= 0
				
				for j in range(self.Items):
					feed_dict = {
								 self.pairwise 		: self.Pairwise[j],
								 self.question 		: self.Question[j],
								 self.answer 		: self.Answer[j],
								 self.review   		: self.Review[j],
								 self.termTotermR 	: self.TermtoTermR[j],
								 self.termTotermP 	: self.TermtoTermP[j],
								 self.question_I 	: self.Question_I[j],
								 self.answer_I 		: self.Answer_I[j],
								 self.review_I 		: self.Review_I[j]
								 }

					train, result 	=  sess.run([train_step, self.loss], feed_dict = feed_dict)
					log_likelihood 	+= result[0]
					accuracy_count 	+= len(result[1][np.where(result[1] > 0.5)])
					count 			+= result[1].size 
					
				accuracy 				= (accuracy_count * 1.0) / count
				valid, test, topRanked 	= self.valid_test_perf(sess)
				
				print "For Training Epoch ", i 
				print "--------------------------------------"
				print "Training Data:"
				print "\tlog_likelihood: ", log_likelihood
				print "\taccuracy: ", accuracy
				print "Valid Data:"
				print "\taccuracy: ", valid
				print "Test Data:"
				print "\taccuracy: ", test 
				print "--------------------------------------"
			
			return sess
