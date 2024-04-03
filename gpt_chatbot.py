#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing dependencies
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import math
import pickle
import argparse


# In[ ]:


parser = argparse.ArgumentParser(description='This is a demonstration program') 

# Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')
args = parser.parse_args()

# Now we can use the argument value in our program.
print(f'batch size: {args.batch_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[ ]:


# Setting up hyperparameters
## Auto-tuning can be implemented to find optimal values of hyperparameters for the system that it is running on
# block [.....] has integers in it, tokenizer for example
# batch size is how many blocks in paralell

block_size = 128
# to use the batch_size cmd arg -> python file_name.py -batch_size 32
batch_size = int(args.batch_size)
# batch_size = 128

learning_rate = 5e-5

# number of features
n_embd = 384 

# number of decoder blocks in transformer architecture
n_layer = 8

# how many features each head captures in paralell
n_head = 8

# defining neurons to discard, 20%
dropout = 0.2 


# In[ ]:


chars = ""
with open("data/vocab.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))

# size of different tokens to use
vocab_size = len(chars)


# In[ ]:


# Tokenizer: encode + decode, assigns numbers to each char
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


# In[ ]:


# Head class: Sealed Dot-product Attention
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")

        # could split up the "* 1 / sqrt(lenght of row) in own cell row for clearer code readibility
        # a_weight = attention weight
        a_weight = q @ k.transpose(-2,-1)# (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        # scaling
        a_weight = a_weight * (1 / math.sqrt(k.shape[-1]))

        # prevent lookahead
        a_weight = a_weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        a_weight = F.softmax(a_weight, dim=-1) # (B, T, T)
        a_weight = self.dropout(a_weight)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = a_weight @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# In[ ]:


# [1, 0, 0]
# [1, 0.6, 0]
# [1, 0.6, 0.4]
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # structuring heads in paralell with a fixed head size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # each head has a unique "feature"
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out


# In[ ]:


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # linear transformation
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # drops out a percentage of neurons, mitigate overfitting
        )

    # forward here means to move it "forward" not forward propagation
    def forward(self, x):
        return self.net(x)


# In[ ]:


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head

        # self attention
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    # post normalizing architecture, previous = previous answer
    def forward(self, x):
        previous = self.sa(x)
        x = self.ln1(x + previous) # layer normal 1
        previous = self.ffwd(x)
        x = self.ln2(x + previous) # layer normal 2
        return x


# In[ ]:


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # lookup table, probability table for next character when given one (aa->zz, all different combinations of two characters)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        # maps indecies to vectors by creating a table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # blocks: number of decoder blocks running sequentially, creating 4 decoder layers
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # normalizing layer, help model converge
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm, can be put in different places for optimization
        # apply a linearity transformation, prepare for Softmax activation function
        self.lm_head = nn.Linear(n_embd, vocab_size)

        
        self.apply(self._init_weights)
        
    # initializing weights within proper range, helping training converge better
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, index, targets=None):
        B, seq_length = index.shape

        tok_emb = self.token_embedding_table(index) # (B,T,C)
        # gives 8 indecies = block_size
        pos_emb = self.position_embedding_table(torch.arange(seq_length, device=device)) # (T,C)

        # Embeddings + positional encoding by embedding and feed them into network (decoding layers)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            # B: batch_size. C: num of channels = vocab_size
            B, seq_length, C = logits.shape
            logits = logits.view(B * seq_length, C)
            targets = targets.view(B * seq_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # generates tokens, index = index for current context
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step, previous single character
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index


# In[ ]:


#creating model with set vocabulry size
model = GPTLanguageModel(vocab_size)

# loading in a pre-trained models parameters
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded succesfully')
# pushing parameters to GPU for faster training
m = model.to(device)


# In[ ]:


# chatbot interface
while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Chatbot:\n{generated_chars}')

