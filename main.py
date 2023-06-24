"""
File 1
In this file we parse the data and create an augmenting tokenizer for our python data.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import BucketIterator, Iterator, Field
from torchtext import data

from tokenize import tokenize, untokenize
import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np
import pandas as pd

import keyword
import random
import math
import time


# Opening the file to read it:
f = open("pycode.txt","r", encoding="utf8")
f_lines = f.readlines()

#print(f_lines[:20])

# Parsing the pycode.txt to readable format of problem statement and solution and storing it in dps
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

# Printing parsed code
"""
i = 0
for dp in dps:
    print("\n question no: ", i+1)
    i+=1
    print(dp['question'][1:])
    print(dp['solution'])
    if i>49:
        break
"""
#print("size: ", len(dps))



# Function to tokenize python code (Python needs different tokenization as 'spacy' can only tokenize english sentences and Python has its own syntax and indentation)
def tokenize_py_code(py_code_string):
    py_tokens = list(tokenize(io.BytesIO(py_code_string.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(py_tokens)):
        tokenized_output.append((py_tokens[i].type, py_tokens[i].string))
    return tokenized_output


# tokenized_sample = tokenize_py_code(dps[1]['solution']) #Tokenizing first datapoint in pycode.txt (sum of two numbers in python)
#print(tokenized_sample)

# Untokenizing the tokenized sample
# untokenized_sample = untokenize(tokenized_sample).decode('utf-8')
#print(untokenized_sample)

# Check the keywords that need to be skipped to avoid hardcoded training of the model
#print(keyword.kwlist)



#Function to augment (randomly mask variables) and tokenize python code -- Use this function instead of tokenize_py_code
def augment_tokenize_py_code(py_code_string, mask_factor = 0.3):

    var_dict = {} #Dict to store masked vars

    # Creating a list for keywords that are not variables and need to be skipped from variable masking
    skip_list = ['range', 'extend', 'enumerate', 'print', 'input', 'ord', 'int', 'float', 'type', 'zip', 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'and', 'or', 'min', 'max']
    skip_list.extend(keyword.kwlist)

    var_counter = 1
    py_tokens = py_tokens = list(tokenize(io.BytesIO(py_code_string.encode('utf-8')).readline))
    tokenized_output = []

    for i in range(0, len(py_tokens)):
        if py_tokens[i].type == 1 and py_tokens[i].string not in skip_list:

            if i>0 and py_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']: #avoiding masking modules
                skip_list.append(py_tokens[i].string)
                tokenized_output.append((py_tokens[i].type, py_tokens[i].string))
            elif py_tokens[i].string in var_dict:                                                   #if variable is already masked
                tokenized_output.append((py_tokens[i].type, var_dict[py_tokens[i].string]))
            elif random.uniform(0,1) > 1-mask_factor:                                               #randomly mask variables that are not masked
                var_dict[py_tokens[i].string] = 'var_' + str(var_counter)
                var_counter += 1
                tokenized_output.append((py_tokens[i].type, var_dict[py_tokens[i].string]))
            else:
                skip_list.append(py_tokens[i].string)
                tokenized_output.append((py_tokens[i].type, py_tokens[i].string))
        
        else:
            tokenized_output.append((py_tokens[i].type, py_tokens[i].string))
    
    return tokenized_output

# tokenized_aug_sample = augment_tokenize_py_code(dps[1]['solution'])
#print ("Masked and tokenized datapoint no. 1: ", tokenized_aug_sample)

# untokenized_aug_sample = untokenize(tokenized_aug_sample).decode('utf-8')
#print ("Masked datapoint no.1: ", untokenized_aug_sample)