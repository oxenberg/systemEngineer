#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torch.nn as nn
from torchtext import data
import torch.optim as optim
import torchtext.vocab as torchvocab
from math import log, isfinite
from collections import Counter,defaultdict
import pandas as pd
import math
import operator
import sys, os, time, platform, nltk, random
import copy
import time

MIN_FREQ = 2
TRAIN_LSTM_FILE = "train.csv"
TEST_LSTM_FILE = "test.csv"
VAL_LSTM_FILE = "val.csv"
INPUT_DIM = 7032 
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 18
N_LAYERS = 2
DROPOUT = 0.25

START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"


allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities

"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed = 1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    #torch.backends.cudnn.deterministic = True

# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'Chen Dahan','name2': 'Miri Yitshaki', 'id1': '204606651','id2': '025144635', 'email1': 'dahac@post.bgu.ac.il','email2': 'miriyi@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
      and emissionCounts data-structures.
      allTagCounts and perWordTagCounts should be used for baseline tagging and
      should not include pseudocounts, dummy tags and unknowns.
      The transisionCounts and emmisionCounts
      should be computed with pseudo tags and shoud be smoothed.
      A and B should be the log-probability of the normalized counts, based on
      transisionCounts and  emmisionCounts

      Args:
        tagged_sentences: a list of tagged sentences, each tagged sentence is a
         list of pairs (w,t), as retunred by load_annotated_corpus().

     Return:
        [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    for sentence in tagged_sentences: 
      # sentence beginning
      prev_tag = START
      for tup in sentence:
        # allTagCounts + perWordTagCounts
        allTagCounts[tup[1]] += 1 
        if tup[0] not in perWordTagCounts:
          perWordTagCounts[tup[0]] = Counter()
        perWordTagCounts[tup[0]][tup[1]] += 1
        # transitionCounts + emissionCounts
        if tup[0] not in emissionCounts:
          emissionCounts[tup[0]] = Counter()
        emissionCounts[tup[0]][tup[1]] += 1
        if prev_tag not in transitionCounts:
          transitionCounts[prev_tag] = Counter()
        transitionCounts[prev_tag][tup[1]] += 1
        prev_tag = tup[1]
      # sentence ending
      if prev_tag not in transitionCounts:
        transitionCounts[prev_tag] = Counter()
      transitionCounts[prev_tag][END] += 1
    smooth_transition_emission(transitionCounts,emissionCounts)
    # normlize transitionCounts + emissionCounts
    # A & B - log-probability of the normalized transitionCounts & emissionCounts
    for i in transitionCounts:
      total = sum(transitionCounts[i].values())
      A[i] = defaultdict(lambda: math.log(sys.float_info.min))
      smooth_len = len(transitionCounts[i]) # laplace
      for key in transitionCounts[i]:
        prob = transitionCounts[i][key]/(total+smooth_len) 
        A[i][key] = math.log(prob) if prob>0 else math.log(sys.float_info.min) 
    for i in emissionCounts:
      total = sum(emissionCounts[i].values())
      B[i] = defaultdict(lambda: math.log(sys.float_info.min)) 
      smooth_len = len(emissionCounts[i]) # laplace
      for key in emissionCounts[i]:
        prob = emissionCounts[i][key]/(total+smooth_len) 
        B[i][key] = math.log(prob) if prob>0 else math.log(sys.float_info.min) 
    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]

def smooth_transition_emission(transitionCounts,emissionCounts):
  """smooth transition and emission matrix

    Args:
    transitionCounts, emissionCounts
  """
  smooth_tag = []
  unique_tags = list(transitionCounts.keys())
  unique_tag_len = len(unique_tags)
  for tag1 in unique_tags:
    for tag2 in unique_tags:
        if tag2 == START:
          continue
        transitionCounts[tag1][tag2] += 1 # add 1 to all, new entries initialized with 1
  for word in emissionCounts:
    for state in emissionCounts[word]:
      emissionCounts[word][state] +=1
    emissionCounts[word][UNK] = 1 # update unknown state for each word
    

def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    tagged_sentence = []
    for word in sentence:
      if word in perWordTagCounts:
        # most frequently associated tag
        tagged_sentence.append((word,max(perWordTagCounts[word].items(), key=operator.itemgetter(1))[0]))
      else:
        # sampling from allTagCounts
        tagged_sentence.append((word,random.choices(list(allTagCounts.keys()),list(allTagCounts.values()))))
    return tagged_sentence

#===========================================
#       POS tagging with HMM
#===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """
    v_last = viterbi(sentence, A,B)
    tag_list = retrace(v_last)
    tagged_sentence = zip(sentence, tag_list)
    return list(tagged_sentence)

# obj to use for backtraking the path 
class ViterbiNode:
  def __init__(self, pos, prev):
    self.pos = pos
    self.prev = prev

  def update_prev(self, prev):
    self.prev = prev

def viterbi(sentence, A,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.
    """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END
    padded_sentences = [START] + sentence + [END]
    viterbi = pd.DataFrame(columns=padded_sentences , index= list(A.keys())) # prob matrix, NXM df ,N- num states , T-sentence len
    viterbi = viterbi.fillna(math.log(sys.float_info.min))
    backpointer = pd.DataFrame(columns=padded_sentences , index= list(A.keys())) # backpointer matrix, NXM df ,N- num states , T-sentence len
    # initialization step - first word in the sentence (START in our case)
    for state in A:
      viterbi.loc[state,padded_sentences[0]] = 0 
      backpointer.loc[state,padded_sentences[0]] = 0
    # recursion step
    for word_idx in range(1,len(padded_sentences)): # for each word update column
      if padded_sentences[word_idx] in B: # words seen in training - no need to consider all tags + currently I'm only considering the A matrix 
        for state in B[padded_sentences[word_idx]]: # row for each state
          if state!=UNK:
            max = math.log(sys.float_info.min) 
            max_state = None
            for prev_state in A:
              temp = viterbi.loc[prev_state][word_idx-1] + A[prev_state][state] + B[padded_sentences[word_idx]][state]
              if temp > max:
                max = temp
                max_state = prev_state
            viterbi.loc[state][word_idx] = max
            backpointer.loc[state][word_idx] = max_state
      else: # OOV
        for state in A: # row for each state
            max = math.log(sys.float_info.min) 
            max_state = None
            for prev_state in A:
              temp = viterbi.loc[prev_state][word_idx-1] + A[prev_state][state] 
              if temp > max:
                max = temp
                max_state = prev_state
            viterbi.loc[state][word_idx] = max
            backpointer.loc[state][word_idx] = max_state
    # create backtraking obj
    v_last = create_backtraking_obj(padded_sentences, viterbi, backpointer)
    return v_last

def create_backtraking_obj(padded_sentences, viterbi, backpointer):
    # helper function for the viterbi alogo - create backtraking obj
    curr_pos = END
    curr_node = None
    prev = None
    for word_idx in reversed(range(len(padded_sentences))):
      min_idx = viterbi.iloc[:,word_idx].idxmax() # min_idx is the pos index 
      curr = ViterbiNode(curr_pos,None)
      curr_pos = backpointer.loc[min_idx][word_idx] 
      if prev: 
        prev.update_prev(curr)
      else:
        v_last = curr
      prev = curr
    return v_last

#a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    tags = []
    curr = end_item
    while curr:
      tags.append(curr.pos)
      curr = curr.prev
    return list(reversed(tags))[1:-1] # remove the Start and End 


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0   # joint log prob. of words and tags
    prev_state = START

    for word,state in sentence:
      if word in B:
        if state in B[word]:
          p += B[word][state]
        else:
          p += B[word][UNK]
      p += A[prev_state][state]
      prev_state = state

    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p


#===========================================
#       POS tagging with BiLSTM
#===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""

# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """

    #load data
    sentences = load_annotated_corpus(params_d['data_fn'])
    prepare_df_lstm(sentences, TRAIN_LSTM_FILE)
    TEXT = data.Field(lower = True)
    UD_TAGS = data.Field(unk_token = None)
    W_CASES = data.Field(unk_token = None)
    fields = [ (None,None),
    ('text',TEXT ), 
    ("udtags", UD_TAGS),
    ("wcases", W_CASES)]
    data_train = data.TabularDataset(path=TRAIN_LSTM_FILE, format='csv', fields=fields,skip_header = True)
    # build the vocabulary - a mapping of tokens to integers.
    TEXT.build_vocab(data_train, min_freq = params_d['min_frequency'], max_size = params_d['max_vocab_size'])
    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'],TEXT.vocab.itos)
    TEXT.vocab.load_vectors(vectors)
    UD_TAGS.build_vocab(data_train)
    W_CASES.build_vocab(data_train)
    # building the model- 
    if params_d['input_rep']:
      model = BiLSTMPOSTaggerCase(params_d['max_vocab_size']+2, 
                          params_d['embedding_dimension'], 
                          HIDDEN_DIM, 
                          params_d['output_dimension'], 
                          params_d['num_of_layers'], 
                          True, 
                          DROPOUT,
                          TEXT.vocab.stoi[TEXT.pad_token])
    else:
      model = BiLSTMPOSTagger(params_d['max_vocab_size']+2, 
                          params_d['embedding_dimension'], 
                          HIDDEN_DIM,
                          params_d['output_dimension'], 
                          params_d['num_of_layers'], 
                          True, 
                          DROPOUT,
                          TEXT.vocab.stoi[TEXT.pad_token]) 

    return {'lstm': model,'input_rep':params_d['input_rep'],'fields':fields}

#no need for this one as part of the API
#def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        'output_dimension': int}
    """

    #TODO complete the code

    #return params_d

def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    vectors = torchvocab.Vectors(name = path ,
                            cache = 'vectors',
                            unk_init = torch.Tensor.normal_)
    return vectors


def train_rnn(model, train_data, val_data = None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    #Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)
    TEXT = dict(model['fields'])['text']
    UD_TAGS = dict(model['fields'])['udtags']
    W_CASES = dict(model['fields'])['wcases']
    model_lstm = model['lstm']
    prepare_df_lstm(train_data, TRAIN_LSTM_FILE)
    train_data = data.TabularDataset(path=TRAIN_LSTM_FILE, format='csv', fields=model['fields'],skip_header = True)
    if not val_data:
      train_data,val_data = train_data.split()
    else:
      prepare_df_lstm(val_data, VAL_LSTM_FILE)
      val_data = data.TabularDataset(path=VAL_LSTM_FILE, format='csv', fields=model['fields'],skip_header = True)
    TAG_PAD_IDX =UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
    #handling the iterator.
    BATCH_SIZE = 128
    train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, val_data), sort_key=lambda x: len(x.text),  batch_size = BATCH_SIZE,  sort_within_batch = True)
    # initialize the weights from a simple Normal distribution        
    model_lstm.apply(init_weights)
    # initialize our model's embedding layer with the pre-trained embedding
    model_lstm.embedding.weight.data.copy_(TEXT.vocab.vectors)
    # initialize the embedding of the pad token to all zeros
    model_lstm.embedding.weight.data[TAG_PAD_IDX] = torch.zeros(model_lstm.embedding.embedding_dim)
    # optimizer, used to update our parameters w.r.t. their gradients
    optimizer = optim.Adam(model_lstm.parameters())
    # define cross-entropy loss, ignore pad token
    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
    # place our model and loss function on our GPU,
    model_lstm = model_lstm.to(device)
    criterion = criterion.to(device)

    #Training the Model
    N_EPOCHS = 10
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model_lstm, train_iterator, optimizer, criterion, TAG_PAD_IDX,model['input_rep'])
        valid_loss, valid_acc = evaluate(model_lstm, valid_iterator, criterion, TAG_PAD_IDX,model['input_rep'])
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_lstm.state_dict(), 'tut1-model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    tokens = [t.lower() for t in sentence]
    model_lstm = model['lstm']
    TEXT = dict(model['fields'])['text']
    UD_TAGS = dict(model['fields'])['udtags']
    W_CASES = dict(model['fields'])['wcases']
    numericalized_tokens = [TEXT.vocab.stoi[t] for t in tokens]
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)
    model_lstm.eval()
    predictions = None
    if model['input_rep']:
      sentence_case = [get_word_case_vector(w) for w in sentence]
      numericalized_cases = [W_CASES.vocab.stoi[t] for t in sentence_case]
      case_tensor = torch.LongTensor(numericalized_cases)
      case_tensor = case_tensor.unsqueeze(-1).to(device)
      predictions = model_lstm(token_tensor,case_tensor)
    else:    
      predictions = model_lstm(token_tensor)
    top_predictions = predictions.argmax(-1)
    tagged_sentence =[]
    predicted_tags = [UD_TAGS.vocab.itos[t.item()] for t in top_predictions]
    for i in range (len(sentence)):
      tagged_sentence.append ((sentence[i] , predicted_tags[i]))

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    model_params = {'max_vocab_size':INPUT_DIM ,
                        'min_frequency':MIN_FREQ,
                        'input_rep': 1,
                        'embedding_dimension': EMBEDDING_DIM,
                        'num_of_layers': N_LAYERS,
                        'output_dimension': OUTPUT_DIM,
                        'pretrained_embeddings_fn': "glove.6B.100d.txt",
                        'data_fn': 'en-ud-train.upos.tsv'}
    return model_params


def get_word_case_vector (word):
  """get_word_case_vector for the case lstm.
    Args:
        word: word(str)
  """
  if word == word.lower():
      word_case = "100"
  elif word == word.upper():
      word_case ="010"
  elif word[0].isupper(): 
      word_case = "001"
  else:
      word_case = "000"
  return word_case

def prepare_df_lstm(data, file_name):
  """Prepare data file for the lstm.

    Args:
        data: list of sentences - (w,t) -  word and tag
        file_name: the file to write the df to
  """
  sentences_list =[]
  tags_list = []
  case_list = []
  for sentence in data:
    sentence_words = ' '.join(p[0] for p in sentence)
    sentence_tags = ' '.join(p[1] for p in sentence)
    sentence_case = ' '.join(get_word_case_vector(w[0]) for w in sentence)
    sentences_list.append(sentence_words)
    tags_list.append(sentence_tags)
    case_list.append(sentence_case)
  # Calling DataFrame constructor after zipping 
  # both lists, with columns specified 
  df = pd.DataFrame(list(zip(sentences_list, tags_list, case_list)), 
                columns =['words', 'tags','wcase'])  
  df.to_csv(file_name)
  return df


class BiLSTMPOSTagger(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout,
                 pad_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
      

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        
        return predictions

class BiLSTMPOSTaggerCase(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.embeddingc = nn.Embedding(input_dim, 3)
        self.lstm = nn.LSTM(embedding_dim+3, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text,caseb):
        embedded = self.embedding(text)
        embeddedc = self.embeddingc(caseb)
        embedded = torch.cat([embedded, embeddedc], dim = 2)
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        
        return predictions

def init_weights(m):
    """init model weights from a simple Normal distribution

    Args:
        m: rnn model
    """
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)


def epoch_time(start_time, end_time):
    """calculate eoch time

    Args:
        start_time, end_time
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def train(model, iterator, optimizer, criterion, tag_pad_idx, case):
    ''' case - 0 for vanilla, 1 for case
    '''
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        text = batch.text
        tags = batch.udtags
        optimizer.zero_grad()
        predictions = None
        if case:
          caseb = batch.wcases
          predictions = model(text,caseb)
        else:
          predictions = model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = criterion(predictions, tags)    
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx, case):
    ''' case - 0 for vanilla, 1 for case
    '''
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags
            if case:
              caseb = batch.wcases
              predictions = model(text,caseb)
            else:
              predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    
    values = model[list(model.keys())[0]]

    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, *values)
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, *values)
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, *values)
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, *values)


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence)==len(pred_sentence)

    correct = 0
    correctOOV = 0 
    OOV = 0 
    for gold,pred in zip(gold_sentence,pred_sentence):
      correct_temp = int(gold[1] == pred[1])
      correct += correct_temp
      if gold[0] not in B:
        OOV += 1
        correctOOV += correct_temp

    return correct, correctOOV, OOV





