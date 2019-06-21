# dataLoad.py
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder


def loadData():
	db = mnistDB()

	#db.loadTrainData()
	#db.loadValidData()
	db.loadTestData()

	return db

def saveData(predict):
	labels_ = []
	for aa in range(len(predict)):
		row = predict[aa]
		maxvalue = -sys.maxsize
		maxid = -1
		for bb in range(len(row)):
			if row[bb] > maxvalue:
				maxvalue = row[bb]
				maxid = bb
		labels_.append(maxid)
	f = open('../Output/sample_submission.csv', 'w+')
	f.writelines('ImageId,Label\n')
	for l in range(len(labels_)):
		string = str(l+1) + ',' + str(labels_[l]) +'\n'
		f.writelines(string)
	f.close()

class mnistDB:
	def __init__(self):
		self.train = minstData()
		self.validation = minstData()
		self.test = minstData()

	def loadTrainData(self):
		train = pd.read_csv('../Data/train.csv')
		self.train.load(train)

	def loadValidData(self):
		validation = pd.read_csv('../Data/validation.csv')
		self.validation.load(validation)

	def loadTestData(self):
		test = pd.read_csv('../Data/test.csv')
		self.test.load(test,False)

class minstData:
	def __init__(self):
		self.images = []
		self.labels = []
		self.num_examples = 0
		self.idx = 0

	def load(self,dt,target = True):
		self.num_examples = len(dt.index)
		
		if target:
			self.images = dt.ix[:,1:].values
			self.labels = self.one_hot(dt.ix[:,0:1].get_values())
		else:
			self.images = dt.ix[:,:].values
			self.labels = [[0 for _ in range(10)] for _ in range(self.num_examples)]

	def one_hot(self,labels):
		encoder = OneHotEncoder(categories='auto')
		_1hot = encoder.fit_transform(labels).toarray()
		return _1hot


	def next_batch(self,batchsize = 100):
		indices = np.random.permutation(self.num_examples)[0:100]
		return self.images[indices], self.labels[indices]
