import gzip
import nltk
import operator
import numpy as np 

from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer 	= RegexpTokenizer(r'\w+')
stop 		= set(stopwords.words('english'))
stemmer 	= SnowballStemmer("english")

class Mapping:
	
	def __init__(self, QAfile, ReviewFile, minReview):

		self.QAfile 		= QAfile
		self.ReviewFile 	= ReviewFile
		self.minReview		= minReview
		self.V 			= 5000
		self.Vocublary  	= defaultdict(int)
		self.WordIDMap  	= {}
		self.RWordIDMap  	= {}
		self.ItemIDMap   	= {}
		self.RItemIDMap  	= {}
		
		

	def create_mappings(self):

		print "Creating Mapping\n"
		print "Reading Question Answer"
		
		ItemsToMaybeKeep 	= {}
		qa 			= gzip.open(self.QAfile, 'r')
		
		for l in qa:
			
			qajson 		= eval(l)
			item 		= qajson['asin']
			
			question    	= tokenizer.tokenize(qajson['question'])
			answer 		= tokenizer.tokenize(qajson['answer'])

			for word in question:
				if stemmer.stem(word) not in stop:
					self.Vocublary[stemmer.stem(word)]+=1
					
			for word in answer:
				if stemmer.stem(word) not in stop:
					self.Vocublary[stemmer.stem(word)]+=1
							
			if item not in ItemsToMaybeKeep:
				ItemsToMaybeKeep[item] = 0
		
		del qa
		print "Read Question Answer\n"

		print "Number of Items for which there exists a questions\n", len(ItemsToMaybeKeep)

		print "Reading Reviews"
		review = gzip.open(self.ReviewFile, 'r')
		
		for l in review:
			
			reviewjson 	= eval(l)
			item 		= reviewjson['asin']
			
			review 		= tokenizer.tokenize(reviewjson['reviewText'])
			
			for word in review:
				if stemmer.stem(word) not in stop:
					self.Vocublary[stemmer.stem(word)]+=1
			
			if item in ItemsToMaybeKeep:
				ItemsToMaybeKeep[item]+=1
		
		del review
		print "Read Reviews\n"

		count 	= 0
		
		for (key,value) in ItemsToMaybeKeep.items():
			if value >= self.minReview:
				self.ItemIDMap[key] 	= count
				self.RItemIDMap[count] 	= key
				count 			= count + 1

		temp_vocab = sorted(self.Vocublary.items(), key = operator.itemgetter(1))
		
		if (len(temp_vocab) > 5000):
			print "Pre-sorting Vocublary length is %d"%(len(temp_vocab))
			temp_vocab = temp_vocab[len(temp_vocab)-5000:]
		
		self.Vocublary 	= dict(temp_vocab)
		self.V 		= len(self.Vocublary)
		count 		= 0
		
		for (key,value) in self.Vocublary.items():
			
			self.WordIDMap[key] 	= count
			self.RWordIDMap[count] 	= key
			count 			= count + 1

		X = sorted(self.Vocublary.items(), key = operator.itemgetter(1))
		
		#Uncomment this for printing out the Vocubalary. It was used during different model testing
		'''
		for i in range(len(X)-1, -1, -1):
			print X[i][0]," : ",X[i][1]	
		'''		
		
		if count != self.V:
			print "\nCount does not equals to Vocublary!!\n"

		print "\nMappings Created\n"

			




				


		
