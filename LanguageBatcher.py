import numpy as np
from pdb import set_trace as T
import os
import nlp, utils
import re
import string

class LanguageBatcher():
   def __init__(self, fName, vocabF, 
         batchSize, seqLen, minContext, rand=False, mode='ptb'):
      self.seqLen = seqLen+1
      self.minContext = minContext
      self.batchSize = batchSize
      self.rand = rand
      self.pos = 0

      self.vocab = utils.loadDict(vocabF)
      self.invVocab = utils.invertDict(self.vocab)

      realLen = self.seqLen - minContext
      if mode == 'french':
          dat = open(fName,encoding = "ISO-8859-1").read()
          dat = dat.lower()
          remove = string.punctuation
          remove = remove.replace("'","")
          pattern = r"[{}]".format(remove)
          dat = re.sub(pattern,"",dat)
          dat = dat[:int(len(dat)/(realLen))*realLen]

      elif mode == 'english_words':
          dat = open(fName).read()
          dat = dat.lower()

      elif mode == 'ptb':
          dat = open(fName).read()
          dat = dat[:int(len(dat)/(realLen))*realLen]

      dat = nlp.applyVocab(dat, self.vocab,mode)
      dat = np.asarray(list(dat))
      if mode =='english_words':
          dat = dat[:int(len(dat)/(realLen))*realLen]

      dat = dat.reshape(-1, realLen)
      dat = np.hstack((dat[:-1], dat[1:, :minContext]))

      self.batches = int(np.floor(dat.shape[0] / batchSize))
      self.dat = dat[:(self.batches*batchSize)]
      self.m = len(self.dat.flat)

   def next(self):
      if not self.rand:
         dat = self.dat[self.pos:self.pos+self.batchSize]
         #s = nlp.applyInvVocab(dat[0], self.vocab)
         self.pos += self.batchSize
         if self.pos >= self.dat.shape[0]:
            self.pos = 0
      else:
         inds = np.random.randint(0, self.m-self.seqLen, (self.batchSize, 1))
         inds = inds + (np.ones((self.batchSize, self.seqLen))*np.arange(self.seqLen)).astype(np.int)
         dat = self.dat.flat[inds]

      return dat[:, :-1], dat[:, 1:]


