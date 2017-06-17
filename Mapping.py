import numpy as np 
from collections import namedtuple,defaultdict
import gzip
import json
import operator
from Helper import disp

class Mapping:
	
	def __init__(self, QAfile, ReviewFile, minReview):

		self.QAfile 	 = QAfile
		self.ReviewFile  = ReviewFile
		self.minReview	 = minReview
		self.V 			 = 5000
		self.Vocublary   = defaultdict(int)
		self.WordIDMap   = {}
		self.RWordIDMap  = {}
		self.ItemIDMap   = {}
		self.RItemIDMap  = {}
		

	def create_mappings(self):

		print "Creating Mapping"

		disp("Reading Question Answer")
		qa = gzip.open(self.QAfile, 'r')
		ItemsToMaybeKeep = {}
		for l in qa:
			qajson = eval(l)
			item = qajson['asin']
			question = [word for sentence in qajson['question'].split('.') for word in sentence.split(' ') if word != '']
			answer	 = [word for sentence in qajson['answer'].split('.') for word in sentence.split(' ') if word != '']
			for word in question:
				self.Vocublary[word]+=1
			for word in answer:
				self.Vocublary[word]+=1
			if item not in ItemsToMaybeKeep:
				ItemsToMaybeKeep[item] = 0
		del qa
		print "Read Question Answer"
		
		disp("Reading Reviews")
		review = gzip.open(self.ReviewFile, 'r')
		for l in review:
			reviewjson = eval(l)
			item = reviewjson['asin']
			review = [word for sentence in reviewjson['reviewText'].split('.') for word in sentence.split(' ') if word != '']
			for word in review:
				self.Vocublary[word]+=1
			if item in ItemsToMaybeKeep:
				ItemsToMaybeKeep[item]+=1
		del review
		print "Read Reviews"

		count = 0
		for (key,value) in ItemsToMaybeKeep.items():
			if value >= self.minReview:
				self.ItemIDMap[key] = count
				self.RItemIDMap[count] = key
				count = count + 1

		temp_vocab = sorted(self.Vocublary.items(), key = operator.itemgetter(1))
		if (len(temp_vocab) > 5000):
			print "Pre-sorting Vocublary length is %d"%(len(temp_vocab))
			temp_vocab = temp_vocab[len(temp_vocab)-5000:]
		self.Vocublary = dict(temp_vocab)
		self.V = len(self.Vocublary)
		count = 0
		for (key,value) in self.Vocublary.items():
			self.WordIDMap[key] = count
			self.RWordIDMap[count] = key
			count = count + 1
		
		if count != self.V:
			print "Count does not equals to Vocublary!!"

		print "Mappings Created"

			




				


		