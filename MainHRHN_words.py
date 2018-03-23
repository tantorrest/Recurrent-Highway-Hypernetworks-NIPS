from pdb import set_trace as T
from Run import run
from models.BaselineLSTM import LSTMCell
from models.RHN import RHNCell
from models.HyperRHN import HyperRHNCell

saveName = 'saves/testme/'
load = False 
test = True
cellFunc = HyperRHNCell
depth = 7
h = 1000
hCell = h
hCell = (128, h)
vocab = 11007
embedDim = 27
batchSz = 200
context = 100
minContext = 50
eta = 1e-3
gateDrop = 1.0 - 0.65
embedDrop = 0.0
#15461 words in train
#2401 words in test
#3061 words in valid

cell = cellFunc(embedDim, hCell, depth, gateDrop)
run(cell, depth, h, vocab, batchSz, embedDim, embedDrop,
      context, minContext, eta, saveName, load, test,mode='english_words')
