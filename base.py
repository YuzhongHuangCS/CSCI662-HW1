import sys
import pdb
from collections import Counter, defaultdict
import numpy as np
import math
from functools import partial
import tempfile
import os

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
		fid, outputfile = tempfile.mkstemp()
		self.test(inputfile, outputfile)
		true_labels = np.asarray([line.rstrip().split('\t')[1] for line in open(inputfile, encoding='utf-8')])
		pred_labels = np.asarray([line.rstrip() for line in open(outputfile, encoding='utf-8')])
		acc = np.mean(true_labels == pred_labels)
		print('Accuracy', acc)

class NaiveBayesModel(Model):
	def __init__(self):
		super(NaiveBayesModel, self).__init__()
		self.label_word_counter = defaultdict(Counter)
		self.total_label_word_counter = Counter()
		self.label_counter = Counter()
		self.label_word_probs = defaultdict(partial(defaultdict, float))
		self.label_probs = defaultdict(float)
		self.vocab = 0

	def train(self, inputfile):
		for line in open(inputfile, 'r', encoding='utf-8'):
			text, label = line.rstrip().split('\t')
			self.label_counter[label] += 1
			for word in text.split():
				self.label_word_counter[label][word] += 1
				self.total_label_word_counter[label] += 1

		total_sentences = sum(self.label_counter.values())
		self.vocab = sum([sum(w.values()) for w in self.label_word_counter.values()])

		for label, count in self.label_counter.items():
			self.label_probs[label] = math.log(count / total_sentences)

			total_word_label = self.total_label_word_counter[label]
			for word in self.label_word_counter[label]:
				self.label_word_probs[label][word] = math.log((self.label_word_counter[label][word]+1) / (total_word_label + self.vocab))

	def test(self, inputfile, outputfile):
		labels = list(self.label_probs.keys())

		with open(outputfile, 'w') as fout:
			for line in open(inputfile, 'r', encoding='utf-8'):
				text = line.rstrip().split('\t')
				#support line with label and without label
				if isinstance(text, list):
					text = text[0]

				scores = []
				for label in self.label_probs:
					total_word_label = self.total_label_word_counter[label]
					score = self.label_probs[label]
					for word in text.split():
						score += self.label_word_probs[label].get(word, math.log(1/(total_word_label + self.vocab)))
					scores.append(score)

				max_score_index = np.argmax(scores)
				pred = labels[max_score_index]
				fout.write(pred + '\n')


def softmax(ary):
	ary_exp = np.exp(ary-np.max(ary))
	return ary_exp / sum(ary_exp)

class PerceptronModel(Model):
	def __init__(self, logistic=False):
		super(PerceptronModel, self).__init__()
		self.label_word_counter = defaultdict(Counter)
		self.total_label_word_counter = Counter()
		self.label_counter = Counter()
		self.label_word_probs = defaultdict(partial(defaultdict, float))
		self.label_probs = defaultdict(float)
		self.vocab = 0
		self.W = None
		self.logistic = logistic

	def train(self, inputfile):
		for line in open(inputfile, 'r', encoding='utf-8'):
			text, label = line.rstrip().split('\t')
			self.label_counter[label] += 1
			for word in text.split():
				self.label_word_counter[label][word] += 1
				self.total_label_word_counter[label] += 1

		total_sentences = sum(self.label_counter.values())
		self.vocab = sum([sum(w.values()) for w in self.label_word_counter.values()])

		for label, count in self.label_counter.items():
			self.label_probs[label] = math.log(count / total_sentences)

			total_word_label = self.total_label_word_counter[label]
			for word in self.label_word_counter[label]:
				self.label_word_probs[label][word] = math.log((self.label_word_counter[label][word]+1) / (total_word_label + self.vocab))


		labels = list(self.label_probs.keys())
		self.label_map = {value: index for index, value in enumerate(labels)}
		self.label_inv_map = {index: value for index, value in enumerate(labels)}

		Y = []
		X = []
		for line in open(inputfile, 'r', encoding='utf-8'):
			text, truth = line.rstrip().split('\t')
			Y.append(self.label_map[truth])

			x = []
			x.append(len(text.split()))
			for index, label in enumerate(self.label_probs):
				score = self.label_probs[label]
				total_word_label = self.total_label_word_counter[label]
				for word in text.split():
					score += self.label_word_probs[label].get(word, math.log(1/total_word_label + self.vocab))
				x.append(score)
			X.append(x)

		raw_X = np.asarray(X)
		raw_Y = np.asarray(Y)
		perm = np.random.permutation(len(X))
		X = raw_X[perm]
		Y = raw_Y[perm]
		W = np.random.uniform(-1, 1, size=(X.shape[1], len(labels)))
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
					truth = np.zeros(len(labels))
					truth[y] = 1
					W += np.matmul(x.reshape(-1, 1), (truth-prob).reshape(1, -1))
				else:
					if y != pred:
						onehot = np.ones(len(labels))
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
				x.append(len(text.split()))
				for index, label in enumerate(self.label_probs):
					score = self.label_probs[label]
					total_word_label = sum(self.label_word_counter[label].values())
					for word in text.split():
						score += self.label_word_probs[label].get(word, math.log(1/total_word_label + self.vocab))
					x.append(score)
				x = np.asarray(x)
				score = np.matmul(x, self.W)
				index = np.argmax(score)
				pred = self.label_inv_map[index]
				fout.write(pred + '\n')
