import time
import sys
import argparse
import random
import copy
import torch
import gc
import cPickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from model.seqmodel import SeqModel_circulationBiLSTM
from model.LSTMText import LSTMText
from utils.data import Data
from utils.data import init_parser
import os

data1 = Data()
data1.load('data/data.substring.base2.pickle')
data2 = Data()
data2.load('data/data.pickle')
print data1
print data2
print 'fff'