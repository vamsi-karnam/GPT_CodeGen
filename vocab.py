"""
In this file we build the vocabulary for seq to seq learning using torchtext fields
"""

import main
import train_val
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch import backends

import torchtext
from torchtext.data import BucketIterator, Iterator, Field
from torchtext import data

SEED = 6174

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
backends.cudnn.deterministic = True

Input = data.Field(tokenize='spacy', init_token='', eos_token='', lower=True)

Output = data.Field(tokenize=main.augment_tokenize_py_code, init_token='', eos_token='', lower=False)

fields = [('Input', Input),('Output', Output)]

"""
The augmented data has the potential to increase vocublary beyond initial data, so we need to capture as many variations as possible
for a wide vocabulary. Say, we augment the dataset 100 or 150 times to capture the majority of the augmented vocublary.
"""

train_example = []
val_example = []

train_expansion_scale = 150

for i in range(train_expansion_scale):                                      #Only expanding the training set
    for j in range(train_val.train_df.shape[0]):
        try:
            ex = data.Example.fromlist([train_val.train_df.question[i], train_val.train_df.solution[i]], fields)
            train_example.append(ex)
        except:
            pass
for i in range(train_val.val_df.shape[0]):
    try:
        ex = data.Example.fromlist([train_val.val_df.question[i], train_val.val_df.solution[i]], fields)
        val_example.append(ex)
    except:
        pass

train_data = data.Dataset(train_example, fields)
val_data = data.Dataset(val_example, fields)

Input.build_vocab(train_data, min_freq=0)
Output.build_vocab(train_data, min_freq=0)
Output.vocab

print(Output.vocab)