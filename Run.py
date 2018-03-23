from pdb import set_trace as T
import sys, shutil
import time
import numpy as np
import torch
import torch as t
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import utils

from LanguageBatcher import LanguageBatcher
from HyperLinear import HyperLinear

#Load PTB
def dataBatcher(batchSz, context, minContext,mode):
	print('Loading Data...')
	if mode == 'french':
		train = 'data/french_corpus.txt'
		valid = 'data/french_corpus_val.txt'
		test  = 'data/french_corpus_test.txt'
		vocab = 'data/vocab_french.txt'
	elif mode == 'english_words':
		train = 'data/english_words_corpus.txt'
		valid = 'data/english_words_val.txt'
		test  = 'data/english_words_test.txt'
		vocab = 'data/reuters_vocab.txt'
	elif mode == 'ptb':
		train = 'data/ptb.train.txt'
		valid = 'data/ptb.valid.txt'
		test  = 'data/ptb.test.txt'
		vocab = 'data/vocab.txt'

	trainBatcher = LanguageBatcher(train, vocab, batchSz, context, 0, rand=True, mode=mode)
	validBatcher = LanguageBatcher(valid, vocab, batchSz, context, minContext,mode=mode)
	testBatcher  = LanguageBatcher(test,  vocab, batchSz, context, minContext,mode=mode)

	print('Data Loaded.')
	return trainBatcher, validBatcher, testBatcher

class Network(nn.Module):

	def __init__(self, cell, vocabDim, embedDim, unembedDim, ansDim, context, embedDrop):
		super(Network, self).__init__()
		self.cell, self.context, self.drop = cell, context, embedDrop
		self.embed   = nn.Embedding(vocabDim, embedDim)
		self.unembed = nn.Linear(unembedDim, ansDim)

	def forward(self, x, trainable):
		x, s, out = self.embed(x), 0, []
		for i in range(self.context):
			o, s, sMetrics = self.cell(x[:, i], s, trainable)
			out += [o]

		batchSz = x.size(0)
		x = t.stack(out, 1).view(batchSz*self.context, -1)
		return self.unembed(x).view(batchSz, self.context, -1)

def train(net, opt, trainBatcher, validBatcher, saver, minContext):
	while True:
		start = time.time()

		#Run epochs
		trainLoss, trainAcc = utils.runData(net, opt, trainBatcher,trainable=True, verbose=True)
		validLoss, validAcc = utils.runData(net, opt, validBatcher, minContext=minContext)
		trainLoss, validLoss = np.exp(trainLoss), np.exp(validLoss)

		#Print statistics
		print('\nEpoch: ', saver.epoch(), ', Time: ', time.time()-start)
		print('| Train Perp: ', trainLoss,', Train Acc: ', trainAcc)
		print('| Valid Perp: ', validLoss,', Valid Acc: ', validAcc)

		if np.isnan(validLoss) or np.isnan(trainLoss):
			print('Got a bad update. Resetting epoch')
			saver.refresh(net)
		else:
			saver.update(net, trainLoss, trainAcc, validLoss, validAcc)

def test(net, batcher, minContext, name='Test'):
	start = time.time()

	loss, acc = utils.runData(net, None, batcher, cuda=False, minContext=minContext)
	loss = np.exp(loss)

	#Print statistics
	print('Time: ', time.time()-start)
	print(name, ' Perp: ', loss, ', Acc: ', acc)

def modelDef(net, cuda=False):
	if cuda: net.cuda()
	utils.initWeights(net)
	utils.modelSize(net)
	return net

def run(cell, depth, h, vocabDim, batchSz, embedDim, embedDrop, context, minContext, eta, saveName, load, isTest,mode):
	trainBatcher, validBatcher, testBatcher = dataBatcher(batchSz, context, minContext,mode)

	net = modelDef(Network(cell, vocabDim, embedDim, h, vocabDim, context, embedDrop))
	opt = t.optim.Adam(net.parameters(), lr=eta)

	saver = utils.SaveManager(saveName)
	if load: saver.load(net)

	if not isTest:
		train(net, opt, trainBatcher, validBatcher, saver, minContext)
	else:
		#Up the test context
		minContext = 95
		test(net, validBatcher, minContext, name='Valid')
		test(net, testBatcher, minContext, name='Test')

