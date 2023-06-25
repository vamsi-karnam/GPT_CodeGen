"""
File 8
Finally, we test the model
"""
# Essential imports
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import BucketIterator, Iterator, Field
from torchtext import data

import time
import math
import pickle

import spacy
from tokenize import tokenize, untokenize

from tqdm import tqdm

# User imports
import main
import train_val
import vocab
import transformer
import training
import loss_fun

# Variables used from user imports
device = main.device
train_df = train_val.train_df
val_df = train_val.val_df
Input = vocab.Input
Output = vocab.Output
fields = vocab.fields
Encoder = transformer.Encoder
Decoder = transformer.Decoder
TRG_PAD_IDX = training.TRG_PAD_IDX
model = training.model
optimiser = training.optimiser
criteria = loss_fun.criteria
argmax = torch.argmax


SRC = Input
TRG = Output

model.load_state_dict(torch.load("/Users/vamsi/Desktop/Learning/My Git/GPT_CodeGen/model/model.pt"))            #Check directory path

def translate(sentence, src_field, trg_field, model, device, max_len=50000):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


#Testing the output:

"""
src = "write a function to subtract two numbers"
src.split(" ")
translation, attention = translate(src, SRC, TRG, model, device)

print("You asked to: ", src)
print("------------------------------------------------------------------------------------------------------------")
print(f'Predicted target (output) sequence: ')
print(translation)
print("------------------------------------------------------------------------------------------------------------")
print("Target sequence translated to code: \n", untokenize(translation[:-1]).decode('utf-8'))
"""