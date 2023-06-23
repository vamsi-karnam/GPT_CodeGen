import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import BucketIterator, Iterator, Field
from torchtext import data

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np
import pandas as pd

import random
import math
import time


#opening the file to read it:
f = open("pycode.txt","r", encoding="utf8")
f_lines = f.readlines()

#print(f_lines[:20])

dps = []
dp = None
for line in f_lines:
    if line[0] == "#":
        if dp:
            dp['solution'] = ''.join(dp['solution'])
            dps.append(dp)
        dp = {"question": None, "solution": []}
        dp['question'] = line[1:]
    else:
        dp['solution'].append(line)

i = 0
for dp in dps:
    print("\n question no: ", i+1)
    i+=1
    print(dp['question'][1:])
    print(dp['solution'])
    if i>49:
        break