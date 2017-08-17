import gzip
import json
import nltk
import numpy as np 

from collections import defaultdict
from nltk.stem import SnowballStemmer	
from utils import check_sent, normalize
from nltk.tokenize import RegexpTokenizer

tokenizer 	= RegexpTokenizer(r'\w+')
stemmer 	= SnowballStemmer("english")

class QAdoc:

	def __init__(self, itemId, questionType, answerType, question, answer, V, WordIDMap):

		self.itemId 		= itemId
		self.questionType 	= questionType
		self.answerType 	= answerType
		self.question 		= question
		self.answer 		= answer
		self.Question 		= [WordIDMap[stemmer.stem(word)] for word in tokenizer.tokenize(question) if stemmer.stem(word) in WordIDMap]
		self.Answer 		= [WordIDMap[stemmer.stem(word)] for word in tokenizer.tokenize(answer) if stemmer.stem(word) in WordIDMap]
		self.qFeature 		= {}
		self.aFeature 		= {}
		self.create_QAFeature()


	def create_QAFeature(self):

		for wordId in self.Question:
			if wordId in self.qFeature:
				self.qFeature[wordId] += 1
			
			else:
				self.qFeature[wordId] = np.float64(1)


		for wordId in self.Answer:
			if wordId in self.aFeature:
				self.aFeature[wordId] += 1
			
			else:
				self.aFeature[wordId] = np.float64(1)
		
		normalize(self.qFeature)
		normalize(self.aFeature)

		

class ReviewDoc:

	def __init__(self, itemId, reviewText, sentences, SPerItem, V, WordIDMap):

		self.itemId 	= itemId
		self.reviewText = reviewText
		self.Review 	= []
		self.create_ReviewFromSentence(sentences, SPerItem, V, WordIDMap)

	def create_ReviewFromSentence(self, sentences, SPerItem, V, WordIDMap):

		review_sentence = self.reviewText.split('.')
		
		for sent in review_sentence:
			if check_sent(sent, WordIDMap):
				obj = Sentence(self.itemId, sent, V, WordIDMap, self)
				sentences.append(obj)
				self.Review.append(len(sentences)-1)
				SPerItem[self.itemId].append(len(sentences)-1)
				obj.create_SFeature()
			

class Sentence: 

	def __init__(self, itemId, Review, V, WordIDMap, ReviewObj):

		self.itemId 	= itemId
		self.sent 	= Review
		self.rObj 	= ReviewObj
		self.Sent 	= [WordIDMap[stemmer.stem(word)] for word in tokenizer.tokenize(Review) if stemmer.stem(word) in WordIDMap]
		self.sFeature 	= {}
				
	def create_SFeature(self):

		for wordId in self.Sent:
			if wordId in self.sFeature:
				self.sFeature[wordId] += 1
			
			else:
				self.sFeature[wordId] = np.float64(1)
		
		normalize(self.sFeature)

		
