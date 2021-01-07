"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
from torchtext import data
import torch.optim as optim
from math import log, isfinite, inf
from collections import Counter
import numpy as np
import sys, os, time, platform, nltk, random

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
    #TODO edit the dictionary to have your own details
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'or oxenberg', 'id1': '312460132', 'email1': 'orox@post.bgu.ac.il',
            'name2': 'sahar baribi', 'id2': '311232730', 'email2': 'saharba@post.bgu.ac.il'}


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
    
    # TODO check if we use the dummy tags right
    transitionCounts[START] = Counter()
    all_words_count = 0
    
    for sentence in tagged_sentences:
        previous_tag = START
        for word,tag in sentence:
            all_words_count +=1
            #: count tags
            allTagCounts[tag] +=1
            
            #: check if the not word in perWordTagCounts dict
            if word not in perWordTagCounts:
                perWordTagCounts[word] = Counter()
            # if previous_tag not in transitionCounts: 
            #     transitionCounts[previous_tag] = Counter()
            if tag not in emissionCounts:
                emissionCounts[tag] = Counter()
                transitionCounts[tag] = Counter()
            #: count tags
            perWordTagCounts[word][tag] +=1
            transitionCounts[previous_tag][tag]+=1
            emissionCounts[tag][word] +=1 
            previous_tag = tag
            
        transitionCounts[previous_tag][END]+=1
        

    
    A = transitionCounts.copy()
    B = emissionCounts.copy()
    
    
    #: filling A
    for tag, dictOfTags in A.items():
        newDictOfTags = {}
        for keyTag in dictOfTags: 
            if tag == START or tag ==END:
                newDictOfTags[keyTag] = log(dictOfTags[keyTag]/len(tagged_sentences))
            elif allTagCounts[tag]==0:
                newDictOfTags[keyTag] = 0
            else:
                newDictOfTags[keyTag] = log(dictOfTags[keyTag]/allTagCounts[tag])
        A[tag] = newDictOfTags
    
    #: filling B
    for tag, dictOfWords in B.items():
        dictOfWords = {k:log(v/allTagCounts[tag]) for k,v in dictOfWords.items()}
        B[tag] = dictOfWords
        B[tag][UNK] = allTagCounts[tag]/all_words_count
    
    #: Handling padding values
    emissionCounts[START] = Counter({START:len(tagged_sentences)})
    emissionCounts[END] = Counter({END:len(tagged_sentences)})   
    # A[START]['<start>'] = 0
    # A[END]['<end>'] = 0
    B[UNK] = {}
    B[UNK]['<UNKNOWN>'] = 0
    
    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]

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
            tag = perWordTagCounts[word].most_common(1)[0][0]
        else:
            counts = np.array(list(allTagCounts.values()))
            counts = counts/sum(counts)
            all_tags = list(allTagCounts.keys())
            tag = np.random.choice(all_tags, p=counts)
        tagged_sentence.append((word,tag))
    
    
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
    viterbi_matrix = viterbi(sentence,A,B)
    tags = retrace(viterbi_matrix)
    tagged_sentence = []
    
    assert len(sentence) == len(tags)
    for i in range(len(sentence)):
        word = sentence[i]
        tag = tags[i]
        tagged_sentence.append((word,tag))

    return tagged_sentence

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

    sentence = [word if word else UNK in perWordTagCounts.keys() for word in sentence] + [END]
    
    states = list(allTagCounts.keys())
    
    viterbi_matrix = []
    for i in range(len(states)):
        row = []
        for j in range(len(sentence)):
            row.append((states[i],0,np.NINF))
        viterbi_matrix.append(row)
        
    viterbi_matrix = asarray(viterbi_matrix)
    # viterbi_matrix = np.array(viterbi_matrix, dtype = [('tag', np.object), ('r', np.int), ('prob', np.float)])
    
    #: initialize first column
    first_word = sentence[0]
    first_column_optional_tags = list(perWordTagCounts[first_word].keys())
    
    for tag in first_column_optional_tags: 
        row_index = states.index(tag)
        b_state_word = B[tag][first_word]
        try:
            transition_probability = A[START][tag]
        except:
            #TODO make sure the probability is correct
            transition_probability = 0
        
        viterbi_matrix[row_index][0] = (tag, START, transition_probability+b_state_word)
    
    for i in range(1, len(sentence)):
        word = sentence[i]
        if word ==UNK: 
            optional_tags = states
        else:
            optional_tags = list(perWordTagCounts[word].keys())
        for tag in optional_tags: 
            row_index = states.index(tag)
            #: find max probability
            for previous_tag_cell in viterbi_matrix[:, i-1]:
                x = A[previous_tag_cell[0]][tag]
                y= previous_tag_cell[2]
            
            # max_value = max([A[previous_tag_cell[0]][tag]+previous_tag_cell[2] for previous_tag_cell in viterbi_matrix[:, i-1]])
            #; find max probability relevant previous state index
            best_state_index = np.argmax([A[previous_tag_cell[0]][tag]+previous_tag_cell[2] for previous_tag_cell in viterbi_matrix[:,i-1]])
            #: calculate viterbi probability value
            probability = max_value+B[tag][word]
            #: fill matrix
            viterbi_matrix[row_index][i] = (tag, best_state_index, probability)
        
    return viterbi_matrix

#a suggestion for a helper function. Not an API requirement
def retrace(viterbi_matrix):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    chosen_tags = []
    row_index = -1
    column_index = -1
    tag = viterbi_matrix[row_index][column_index][0]
    while tag!=START:
        row_index = viterbi_matrix[row_index][column_index][1]
        column_index -= 1
        tag = viterbi_matrix[row_index][column_index][0]

        chosen_tags.append(tag)
    
    #:remove the start token and reverse the list
    chosen_tags = chosen_tags[:-1]
    chosen_tags.reverse()
    
    return chosen_tags

#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tupple)
    """


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0   # joint log prob. of words and tags

    sentence = [(START, START)] + [(word, tag) if word else (UNK, tag) in perWordTagCounts.keys() for word,tag in sentence] + [(END,END)]
    previous_tag = sentence[0][1]
    for word, tag in sentence[1:]:
        if tag in A[previous_tag]:
            p+=A[previous_tag][tag]
        p+=B[tag][word]
        previous_tag = tag

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
    """Returns an lstm model based on the specified parameters.

    Args:
        params_d (dict): an dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'input_dimension': int,
                        'embedding_dimension': int,
                        'num_of_layers': int,
                        'output_dimension': int}
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        torch.nn.Module object
    """

    #TODO complete the code

    return model

def get_model_params(model):
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

    return params_d

def load_pretrained_embeddings(path):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.
    """
    #TODO
    return vectors


def train_rnn(model, data_fn, pretrained_embeddings_fn, input_rep = 0):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (torch.nn.Module): the model to train
        data_fn (string): full path to the file with training data (in the provided format)
        pretrained_embeddings_fn (string): full path to the file with pretrained embeddings
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    #Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider loading the data and preprocessing it
    # 4. consider using batching
    # 5. some of the above could be implemented in helper functions (not part of
    #    the required API)

    #TODO complete the code

    criterion = nn.CrossEntropyLoss() #you can set the parameters as you like
    vectors = load_pretrained_embeddings(pretrained_embeddings_fn)

    model = model.to(device)
    criterion = criterion.to(device)


def rnn_tag_sentence(sentence, model, input_rep = 0):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence. Tagging is done with the Viterby
        algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (torch.nn.Module):  a trained BiLSTM model
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful

    Return:
        list: list of pairs
    """

    #TODO complete the code

    return tagged_sentence

def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    #TODO complete the code

    return model_params


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
        3. Vanilla BiLSTM: {'blstm':[Torch.nn.Module, input_rep]}
        4. BiLSTM+case: {'cblstm': [Torch.nn.Module, input_rep]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM:
        the neural network model
        input_rep (int) - must support 0 and 1 (vanilla and case-base, respectively)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0]=='baseline':
        return baseline_tag_sentence(sentence, model.values()[0], model.values()[1])
    if list(model.keys())[0]=='hmm':
        return hmm_tag_sentence(sentence, model.values()[0], model.values()[1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, model.values()[0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, model.values()[0])

def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    correct = 0
    correctOOV = 0
    OOV = 0
    assert len(gold_sentence)==len(pred_sentence)
    
    for gold_tuple, pred_tuple in zip(gold_sentence, pred_sentence): 
        word = gold_tuple[0]
        if gold_tuple[1]==pred_tuple[1]:
            correct +=1            
            if word not in perWordTagCounts: 
                correctOOV +=1
        else: 
            if word not in perWordTagCounts: 
                OOV +=1

    return correct, correctOOV, OOV
