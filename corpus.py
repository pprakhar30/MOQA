import sys
import gzip
import json
import operator
import numpy as np 

from math import log
from mapping import Mapping
from utils import calc_LCS
from collections import defaultdict
from documents import QAdoc, ReviewDoc, Sentence


class Corpus:

	def __init__(self, QAFile, ReviewFile, minReview):

		self.Map 				= Mapping(QAFile, ReviewFile, minReview)
		self.QAnswers  				= []
		self.Sentences 				= []
		self.QPerItem  				= []
		self.SPerItem  				= []
		self.PairWiseFeature 			= {}
		self.Avgdl 				= defaultdict(float)
		
		self.Map.create_mappings()
		
		for i in range(len(self.Map.ItemIDMap)):
			self.QPerItem.append([])
		
		for i in range(len(self.Map.ItemIDMap)):
			self.SPerItem.append([])

	
	def construct_QAnswersAndQPerItem(self):
		
		print "Creating Question Answer objects\n"
		print "Reading QA Files"
		
		qa = gzip.open(self.Map.QAfile, 'r')
		
		for qajson in qa:
			l = eval(qajson)
			
			if l['asin'] in self.Map.ItemIDMap.keys():
				itemId   = self.Map.ItemIDMap[l['asin']]
				qType    = l['questionType']
				
				if qType == 'open-ended':
					aType = 'Not Applicable'
				
				else:
					aType = l['answerType']
				
				question = l['question']
				answer   = l['answer']
				obj      = QAdoc(itemId, qType, aType, question, answer, self.Map.V, self.Map.WordIDMap)
				
				self.QAnswers.append(obj)
				self.QPerItem[itemId].append(len(self.QAnswers)-1)
		
		del qa
		print "Read QAfiles\n"
	

	def construct_SentencesAndSPerItem(self):
		
		print "Creating Sentences per Review\n"
		print "Reading Review Files"
		
		review = gzip.open(self.Map.ReviewFile, 'r')
		
		for rjson in review:
			l = eval(rjson)

			if l['asin'] in self.Map.ItemIDMap:
				itemID 		= self.Map.ItemIDMap[l['asin']]
				reviewText  	= l['reviewText']
				obj    		= ReviewDoc(itemID, reviewText, self.Sentences, self.SPerItem, self.Map.V, self.Map.WordIDMap)
		
		del review
		print "Read Reviews\n"
	

	def Calculate_PairWiseFeature(self):

		print "Creating Pairwise Feature\n"
		k1 			= 1.2
		b  			= 0.75
		json_dict 		= {}
		
		for itemID in range(len(self.Map.ItemIDMap)):
			IDF,TF,avgdl 		= self.helper(itemID)
			self.Avgdl[itemID] 	= avgdl
			
			for question in self.QPerItem[itemID]:
				for sent in self.SPerItem[itemID]:
					bm25 		= 0.0
					bm25_plus 	= 0.0
					
					for wordID in self.QAnswers[question].Question:
						numr 		= IDF[wordID] * TF[sent,wordID]*(k1 + 1)
						denr 		= TF[sent,wordID] + k1*(1 - b + (b * len(self.Sentences[sent].Sent)/avgdl))
						bm25 		+= (numr*1.0)/denr
						bm25_plus 	+= bm25 + IDF[wordID]
					
					LCS 									= calc_LCS(self.QAnswers[question].Question, self.Sentences[sent].Sent)
					self.PairWiseFeature[(question,sent)] 	= np.array([[bm25, bm25_plus, LCS]], dtype =np.float64)
								
			print "Done for item %d"%(itemID)

		print "Created Pairwise Features\n"

		
	def helper(self, itemID):

		N   	= len(self.SPerItem[itemID])
		TF  	= defaultdict(int)
		IDF 	= np.zeros((self.Map.V))
		DF  	= defaultdict(int)
		avgdl 	= 0.0

		for sent in self.SPerItem[itemID]:
			Sent 	= self.Sentences[sent]
			avgdl 	+= len(Sent.Sent)
			
			for wordID in Sent.Sent:
				TF[sent,wordID]		+=1
				DF[wordID,sent] 	= 1

		for  wordID in range(0,self.Map.V):
			nt = sum([1 for ID,sent in DF if wordID == ID])
			
			if nt != 0:
				IDF[wordID] = log(N+1) - log(nt)

		avgdl = (avgdl*1.0)/N

		return IDF,TF,avgdl

