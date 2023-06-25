"""
File 7
Training and saving the model (~50 epochs)
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


def make_trg_mask(trg):
    trg_pad_mask = (trg!=TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

def train(model, iterator, optimiser, criterion, clip):
    model.train()

    n_total = 0
    print_losses = []

    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        loss = 0
        src = batch.Input.permute(1,0)
        trg = batch.Output.permute(1,0)
        trg_mask = make_trg_mask(trg)
        optimiser.zero_grad()

        output, _ = model(src, trg[:,:-1])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1,output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        mask_loss, nTotal = criterion(output, trg, trg_mask)
        mask_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimiser.step()

        print_losses.append(mask_loss.item()*nTotal)
        n_total += nTotal
    
    return sum(print_losses) / n_total


def evaluate(model,iterator,criterion):
    model.eval()

    n_total = 0
    print_losses = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

            src = batch.Input.permute(1,0)
            trg = batch.Output.permute(1,0)
            trg_mask = make_trg_mask(trg)

            output, _ = model(src, trg[:,:-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1,output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            mask_loss, nTotal = criterion(output,trg,trg_mask)

            print_losses.append(mask_loss.item()*nTotal)
            n_total += nTotal
    
    return sum(print_losses) / n_total


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 50
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_example = []
    val_example = []

    for i in range(train_df.shape[0]):
        try:
            ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
            train_example.append(ex)
        except:
            pass
    
    for i in range(val_df.shape[0]):
        try:
            ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
            val_example.append(ex)
        except:
            pass
    
    train_data = data.Dataset(train_example, fields)
    val_data = data.Dataset(val_example, fields)

    BATCH_SIZE = 16
    train_iterator, val_iterator = BucketIterator.splits((train_data, val_data), batch_size=BATCH_SIZE, sort_key = lambda x: len(x.Input), sort_within_batch=True, device = device)

    train_loss = train(model, train_iterator, optimiser, criteria, CLIP)
    val_loss = evaluate(model, val_iterator, criteria)

    end_time = time.time()

    epoch_mins, epoch_Secs = epoch_time(start_time, end_time)

    if val_loss<best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), "/Users/vamsi/Desktop/Learning/My Git/GPT_CodeGen/model/model.pt") #Change directory field before running
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_Secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train perplexity: {math.exp(train_loss):7.3f}')
    print(f'\tValidation Loss: {val_loss:.3f} | Validation perplexity: {math.exp(val_loss):7.3f}')

    """
    Note: Perplexity is defined as the exponentiated average negative log-likelihood of a tokenized sequence.
          This measure is usually used for casual language models to determine the context of prediction.
          The more the perplexity, the more context the model will have in making the prediction.
          GPT-2 perplexity = ~19.93
    """