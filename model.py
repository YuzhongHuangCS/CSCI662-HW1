import sys
import pdb
from collections import Counter, defaultdict
import numpy as np
import math
from functools import partial
import tempfile
import os
from feature import Feature

def softmax(ary):
	ary_exp = np.exp(ary-np.max(ary))
	return ary_exp / sum(ary_exp)

class Model(object):
	def __init__(self):
		super(Model, self).__init__()

	@classmethod
	def create(cls, name):
		if name == 'naivebayes':
			return NaiveBayesModel()
		elif name == 'perceptron':
			return PerceptronModel()
		elif name == 'logistic':
			return PerceptronModel(logistic=True)
		else:
			print('Unrecognized model name')
			exit()

	def train(self, inputfile):
		pass

	def test(self, inputfile, outputfile):
		pass

	def score(self, inputfile):
		all_lines = open(inputfile, encoding='utf-8').readlines()
		np.random.shuffle(all_lines)

		n_total = len(all_lines)
		n_test = int(n_total * 0.1)
		n_train = n_total - n_test

		fid, train_inputfile = tempfile.mkstemp()
		fid, test_inputfile = tempfile.mkstemp()
		fid, test_outpufile = tempfile.mkstemp()

		with open(train_inputfile, 'w', encoding='utf-8') as fout:
			for line in all_lines[:n_train]:
				fout.write(line)

		with open(test_inputfile, 'w', encoding='utf-8') as fout:
			for line in all_lines[n_train:]:
				fout.write(line)

		self.train(train_inputfile)
		self.test(test_inputfile, test_outpufile)

		true_labels = np.asarray([line.rstrip().split('\t')[1] for line in open(test_inputfile, encoding='utf-8')])
		pred_labels = np.asarray([line.rstrip() for line in open(test_outpufile, encoding='utf-8')])
		acc = np.mean(true_labels == pred_labels)
		print('Accuracy', acc)

class NaiveBayesModel(Model):
	def __init__(self):
		super(NaiveBayesModel, self).__init__()

	def train(self, inputfile):
		self.f = Feature.create('ConditionalProbability', inputfile)

	def test(self, inputfile, outputfile):
		with open(outputfile, 'w') as fout:
			for line in open(inputfile, 'r', encoding='utf-8'):
				text = line.rstrip().split('\t')
				#support line with label and without label
				if isinstance(text, list):
					text = text[0]

				scores = self.f.process(text)
				max_score_index = np.argmax(scores)
				pred = self.f.labels[max_score_index]
				fout.write(pred + '\n')

class PerceptronModel(Model):
	def __init__(self, features=['ConditionalProbability', 'Length'], logistic=False):
		super(PerceptronModel, self).__init__()
		self.W = None
		self.features = features
		self.logistic = logistic

	def train(self, inputfile):
		Y = []
		X = []

		#even if we don't use ConditionalProbability feature, need to get label info from it
		self.fs = [Feature.create(f, inputfile) for f in self.features]
		if 'ConditionalProbability' in self.features:
			self.f = self.fs[self.features.index('ConditionalProbability')]
		else:
			self.f = Feature.create('ConditionalProbability', inputfile)

		for line in open(inputfile, 'r', encoding='utf-8'):
			text, truth = line.rstrip().split('\t')
			Y.append(self.f.label_map[truth])

			x = []
			for f in self.fs:
				x += f.process(text)

			X.append(x)

		raw_X = np.asarray(X)
		raw_Y = np.asarray(Y)
		perm = np.random.permutation(len(X))
		X = raw_X[perm]
		Y = raw_Y[perm]

		W = np.random.uniform(-1, 1, size=(X.shape[1], len(self.f.labels)))
		n_total = len(X)
		n_valid = int(n_total  * 0.1)
		n_train = n_total - n_valid

		X_train = X[:n_train]
		Y_train = Y[:n_train]
		X_valid = X[n_train:]
		Y_valid = Y[n_train:]

		lr = 0.1
		wait = 0
		best_valid_acc = -float('inf')
		best_W = None
		n_decay = 5
		n_wait = 20

		for epoch in range(100):
			train_acc = []
			valid_acc = []
			for i, x in enumerate(X_train):
				score = np.matmul(x, W)
				pred = np.argmax(score)
				y = Y_train[i]
				train_acc.append(pred == y)

				if self.logistic:
					prob = softmax(score)
					truth = np.zeros(len(self.f.labels))
					truth[y] = 1
					W += np.matmul(x.reshape(-1, 1), (truth-prob).reshape(1, -1))
				else:
					if y != pred:
						onehot = np.ones(len(self.f.labels))
						onehot[y] = -1
						W -= lr * np.matmul(x.reshape(-1, 1), onehot.reshape(1, -1))

			for i, x in enumerate(X_valid):
				score = np.matmul(x, W)
				pred = np.argmax(score)
				y = Y_valid[i]
				valid_acc.append(pred == y)

			train_acc = np.mean(train_acc)
			valid_acc = np.mean(valid_acc)
			print(epoch, train_acc, valid_acc)
			if valid_acc > best_valid_acc:
				best_valid_acc = valid_acc
				best_W = W
				wait = 0
			else:
				wait += 1
				print(wait)
				if wait % n_decay == 0:
					lr *= 0.95
					print('new lr', lr)
				if wait >= n_wait:
					W = best_W
					break

		self.W = W

	def test(self, inputfile, outputfile):
		with open(outputfile, 'w') as fout:
			for line in open(inputfile, 'r', encoding='utf-8'):
				text = line.rstrip().split('\t')
				#support line with label and without label
				if isinstance(text, list):
					text = text[0]

				x = []
				for f in self.fs:
					x += f.process(text)

				x = np.asarray(x)
				score = np.matmul(x, self.W)
				index = np.argmax(score)
				pred = self.f.label_inv_map[index]
				fout.write(pred + '\n')
