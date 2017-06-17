import numpy as np 
import dill
from Corpus import Corpus
from Model_I import Model
import sys
import os 
import psutil

if __name__ == "__main__":

	corpus = None

	#if (len(sys.argv)>3):
		
	QAfile 	   = sys.argv[1]
	ReviewFile = sys.argv[2]
	minReview  = int(sys.argv[3])
	k		   = int(sys.argv[4])
	numiter	   = int(sys.argv[5])
	
	corpus	   = Corpus(QAfile, ReviewFile, minReview)
	corpus.construct_QAnswersAndQPerItem()
	corpus.construct_SentencesAndSPerItem()
	corpus.Calculate_PairWiseFeature()

	
	
	model	   = Model(k, numiter, corpus)
	print "Vocabulary Size: "+str(model.V)
	print "Number of Questions: "+str(len(corpus.QAnswers))
	print "Number of Reviews: "+str(len(corpus.Sentences))
	print "Number of Items "+str(model.Items)
	print "Avg review length "+str(sum(corpus.Avgdl.values())/len(corpus.Avgdl))

	model.train_model()
	model.load_model()

	params 										= [model.theta, model.RelvPar, model.A, model.B, model.PredPar, model.X, model.Y]
	valid_accuracy, test_accuracy, topRanked 	= model.valid_test_perf()
	valid_AUC, test_AUC 						= model.AUC()

	print "-----------------------------------------------"
	print "----------------------------------------------\n"
	print "Accuracy: "
	print "\tValidation: "+str(valid_accuracy)
	print "\tTest: "+str(test_accuracy)
	print "\n"
	print "AUC: "
	print "\tValidation: "+str(valid_AUC)
	print "\tTest: "+str(test_AUC)
	print "\n"
	print "-----------------------------------------------"
	print "----------------------------------------------\n"

	if (predictionsOut):
    	model.save_predictions(topRanked, predictionsOut)

    if (modelOut):
    	model.save_model(modelOut)

    if (rankingOut):

    	topRanked = model.top_ranked(10)
    	model.save_top_ranked(topRanked, rankingOut)


		


																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																							
'''print "Creating pickle"
with open('corpus.pkl', 'wb') as output:
	dill.dump(corpus, output)
print "Created pickle"
process = psutil.Process(os.getpid())
print(process.memory_info()[0]/2.**30)
print(process.memory_info().vms)
print(process.memory_info().rss)
print(psutil.cpu_percent())
print(psutil.virtual_memory())
print(process.memory_full_info())
print(process.memory_percent())	

total = 0
	maxQ = -1
	maxS = -1

	for item, itemId in corpus.Map.ItemIDMap.items():
		q = len(corpus.QPerItem[itemId])
		s = len(corpus.SPerItem[itemId])
		print q, s, corpus.Avgdl[itemId]
		maxQ = q if maxQ < q else maxQ
		maxS = s if maxS < s else maxS
		total += q*s*3 + 12*q*s*corpus.Avgdl[itemId] + 12*q*corpus.Avgdl[itemId] + 1*s*corpus.Avgdl[itemId]   
	print "---------------------------"
	print maxQ, maxS
	print total, (total*4.0/10**9)
	print "-----------------------------"

else:
	
	print "Loading pickle"
	with open('corpus.pkl', 'rb') as input:
		corpus 	= dill.load(input)
	print "Loaded pickle"
	k		= int(sys.argv[1])
	numiter	= int(sys.argv[2])
	model 	= Model(k, numiter, corpus)	

	model.train_model()
'''



