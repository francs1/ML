# dataLoad.py
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder


def preprecessData():
	train = pd.read_csv('train.csv')
	train = train / 255
	train.to_csv('train.csv',index=False)

	validation = pd.read_csv('validation.csv')
	validation = train / 255
	validation.to_csv('validation.csv',index=False)

	test = pd.read_csv('test.csv')
	test = train / 255
	test.to_csv('test.csv',index=False)



def loadData():
	db = mnistDB()

	db.loadTrainData()
	db.loadValidData()
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
	f = open('sample_submission.csv', 'w+')
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
		train = pd.read_csv('train.csv')
		self.train.load(train)

	def loadValidData(self):
		validation = pd.read_csv('validation.csv')
		self.validation.load(validation)

	def loadTestData(self):
		test = pd.read_csv('test.csv')
		self.test.load(test,False)

class minstData:
	def __init__(self):
		self.images = np.array([])
		self.labels = np.array([])
		self.nextimg = np.array([])
		self.nextlab = np.array([])
		self.num_examples = 0
		self.idx = 0
		self.indices = []
		self.gp = 0


	def load(self,dt,target = True):
		self.num_examples = len(dt.index)
		self.indices = np.random.permutation(self.num_examples)
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
		size = batchsize * self.gp % self.num_examples
		self.gp += 1 
		#print(size,size+100)
		self.nextimg = self.images[size:size+100]
		self.nextlab = self.labels[size:size+100]
