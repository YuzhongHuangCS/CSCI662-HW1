import sys
import pdb
from collections import Counter, defaultdict
import numpy as np
import math
from functools import partial
import tempfile
import os

class Feature(object):
	def __init__(self, inputfile):
		super(Feature, self).__init__()
		self.ngram = [1]

	@classmethod
	def create(cls, name, inputfile):
		if name == 'ConditionalProbability':
			return ConditionalProbabilityFeature(inputfile)
		elif name == 'Length':
			return LengthFeature(inputfile)
		else:
			print('Unrecognized feature name')
			exit()

	def find_ngrams(self, input_list, n):
		pairs = list(zip(*[input_list[i:] for i in range(n)]))
		return ['_'.join(x) for x in pairs]

	def readline(self, line):
		ret = []
		parts = line.split()
		for n in self.ngram:
			ret += self.find_ngrams(parts, n)

		return ret

	def process(self, line):
		pass

class ConditionalProbabilityFeature(Feature):
	def __init__(self, inputfile):
		super(ConditionalProbabilityFeature, self).__init__(inputfile)
		self.label_word_counter = defaultdict(Counter)
		self.total_label_word_counter = Counter()
		self.label_counter = Counter()
		self.label_word_probs = defaultdict(partial(defaultdict, float))
		self.label_probs = defaultdict(float)
		self.vocab = 0

		for line in open(inputfile, 'r', encoding='utf-8'):
			text, label = line.rstrip().split('\t')
			self.label_counter[label] += 1
			for word in self.readline(text):
				self.label_word_counter[label][word] += 1
				self.total_label_word_counter[label] += 1

		total_sentences = sum(self.label_counter.values())
		self.vocab = sum([sum(w.values()) for w in self.label_word_counter.values()])

		for label, count in self.label_counter.items():
			self.label_probs[label] = math.log(count / total_sentences)

			total_word_label = self.total_label_word_counter[label]
			for word in self.label_word_counter[label]:
				self.label_word_probs[label][word] = math.log((self.label_word_counter[label][word]+1) / (total_word_label + self.vocab))


		self.labels = list(self.label_probs.keys())
		self.label_map = {value: index for index, value in enumerate(self.labels)}
		self.label_inv_map = {index: value for index, value in enumerate(self.labels)}

	def process(self, line):
		x = []
		for label in self.label_probs:
			total_word_label = self.total_label_word_counter[label]
			score = self.label_probs[label]
			for word in self.readline(line):
				score += self.label_word_probs[label].get(word, math.log(1/(total_word_label + self.vocab)))
			x.append(score)
		return x

class LengthFeature(Feature):
	def __init__(self, inputfile):
		super(LengthFeature, self).__init__(inputfile)

	def process(self, line):
		return [len(self.readline(line))]
