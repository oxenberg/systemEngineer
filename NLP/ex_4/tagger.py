"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from math import log, isfinite, inf
from collections import Counter
import numpy as np
import sys, os, time, platform, nltk, random
import pickle 

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


#TODO remove before submission
create_vectors = False

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

    sentence = [word if word in perWordTagCounts.keys() else UNK for word in sentence] + [END]
    
    states = list(allTagCounts.keys()) 
    
    viterbi_matrix = []
    for i in range(len(states)):
        row = []
        for j in range(len(sentence)-1):
            row.append((states[i],0,np.NINF))
        viterbi_matrix.append(row)
        
    
    #: initialize first column
    first_word = sentence[0]
    if first_word == UNK: 
        first_column_optional_tags = states
    else: 
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
    
    for i in range(1, len(sentence[:-1])):
        word = sentence[i]
        if word == UNK: 
            optional_tags = states
        else:
            try:
                optional_tags = list(perWordTagCounts[word].keys())
            except:
                print('b')
        for tag in optional_tags: 
            row_index = states.index(tag)
            
            # : cut and transpose matrix
            t_viterbi_matrix = list(zip(*viterbi_matrix))
            t_viterbi_matrix_cut = t_viterbi_matrix[:i]
            
            tag, best_state_index, probability = predict_next_best(word,tag,t_viterbi_matrix_cut,A,B)
            
            viterbi_matrix[row_index][i] = (tag, best_state_index, probability)
            
    #add the end to calculation
    t_viterbi_matrix = list(zip(*viterbi_matrix))
    # t_viterbi_matrix_cut = t_viterbi_matrix[:i+1]
    tag, best_state_index, probability = predict_next_best(END,END,t_viterbi_matrix,A,B)
    viterbi_matrix.append([(tag, best_state_index, probability)])
        
    return viterbi_matrix

#a suggestion for a helper function. Not an API requirement
def retrace(viterbi_matrix):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    chosen_tags = []
    column_index = -1
    row_index = viterbi_matrix[-1][column_index][1]
    viterbi_matrix = viterbi_matrix[:-1]
    while row_index!=START:

        # print(f"row,column : {(row_index,column_index)}")
        # print(f"item: {viterbi_matrix[row_index][column_index]}")
        tag = viterbi_matrix[row_index][column_index][0]
        # print(f"tag {tag}")
        row_index = viterbi_matrix[row_index][column_index][1]
        column_index -= 1

        chosen_tags.append(tag)
    
    #:remove the start token and reverse the list
    # chosen_tags = chosen_tags[]
    chosen_tags.reverse()
    
    return chosen_tags

#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, viterbi_matrix,A,B):
    """Returns a new item (tupple)
    """
     #add the end to calculation
    
    new_list = []
    for previous_tag_cell in viterbi_matrix[-1]:
        try:
            transision_proba = A[previous_tag_cell[0]][tag]
        except: #: we didn't see tag_t after tag_t-1
            transision_proba = 1/sum([v for v in A[previous_tag_cell[0]].values()])
        previous_vitarbi_path = previous_tag_cell[2]

        new_list.append(previous_vitarbi_path+transision_proba)
    max_value = max(new_list)
    #; find max probability relevant previous state index
    best_state_index = np.argmax(new_list)
    
    if tag == END:
        probability = max_value
    else:
        probability = max_value+B[tag][word]

    
    return (tag,best_state_index, probability)

#TODO check that this function works
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

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, vocab_size, tagset_size, num_layers, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.input_dim = vocab_size
        self.output_dim = tagset_size
        self.cased_flag = False

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        self.word_to_ix = {}
        self.tag_to_ix = {}


    def forward(self, sentence, case_based_vectors = None):
        embeds = self.word_embeddings(sentence) 
        if self.cased_flag:
            embeds = torch.cat([embeds, case_based_vectors], dim = 1)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim = 1)
        
        return tag_scores
    
    def load_embeddings(self, weights_matrix, non_trainable = True):
         self.word_embeddings.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
         if non_trainable:
              self.word_embeddings.weight.requires_grad = False
              
    def change_to_case_based(self):
        self.cased_flag = True
        self.lstm = nn.LSTM(self.embedding_dim+3, self.hidden_dim, self.num_layers, bidirectional=True)

        

    

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
    # TODO check if I can assume the dictionary will include all the values I want
    params_d['hidden_dim'] = 2
    
    model = LSTMTagger(params_d['embedding_dimension'],params_d['input_dimension'],params_d['output_dimension'], params_d['num_of_layers'],params_d['hidden_dim'])  

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
    params_d = {'input_dimension':model.input_dim, 
                'embedding_dimension':model.embedding_dim, 
                'num_of_layers': model.hidden_dim, 
                'output_dimension': model.output_dim}

    return params_d

# TODO check the function
def load_pretrained_embeddings(path):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.
    """
    vectors = {}
    
    with open(path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            vectors[word]=vect

    return vectors


def case_based_function(word):
    if word.islower():
        vec = np.array([1,0,0])
    elif word.isupper():
        vec = np.array([0,1,0])
    else:
        vec = np.array([0,0,1])
    return vec

def create_case_based_vectors(data):
    #: vector will be in the format (fullLower, fullUpper, combined)
    words_case_vector_dict = {}
    for sentence in data:
        for word,tag in sentence:    
            vec = case_based_function(word)
            words_case_vector_dict[word.lower()] = vec
    return words_case_vector_dict


def process_data(data):
    word_to_ix = {}
    tag_to_ix = {}
    training_data = []
    
    for sentence in data:
        words = []
        tags = []
        for word,tag in sentence:
            words.append(word.lower())
            tags.append(tag)
            if word not in word_to_ix:
                word_to_ix[word.lower()] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
        training_data.append((words, tags))
    
    word_to_ix[UNK] = len(word_to_ix)
    vocab_size = len(word_to_ix)
    tags_size = len(tag_to_ix)  
    
    return word_to_ix, tag_to_ix, training_data

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
    data = load_annotated_corpus(data_fn)
 
    if input_rep == 1:
        words_case_vector_dict = create_case_based_vectors(data)    
        #TODO make sure it doesn't mess up the weighted matrix
        model.change_to_case_based()
    
    #: creating the word_to_ix and tag_to_ix dicts 
    #: and converting data to training data - change the list of tuples to a tuple of lists 
    word_to_ix, tag_to_ix, training_data = process_data(data)
    
    criterion = nn.CrossEntropyLoss() #you can set the parameters as you like
    
    #TODO remove before submission. leave only the line that creates the vector
    if create_vectors:
        vectors = load_pretrained_embeddings(pretrained_embeddings_fn)    
        pickle.dump(vectors, open('embeddings/vectors.pkl', 'wb'))
    else:
        vectors = pickle.load(open('embeddings/vectors.pkl', 'rb'))
    
    weighted_matrix = create_weighted_matrix(word_to_ix, vectors, emb_dim = model.embedding_dim)
    model.load_embeddings(weighted_matrix)
    
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.word_to_ix = word_to_ix
    model.tag_to_ix = tag_to_ix
    
    
    for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
    
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
    
            # Step 3. Run our forward pass.
            if input_rep == 1:
                case_based_vectors = torch.tensor([words_case_vector_dict[word] for word in sentence])
                tag_scores = model(sentence_in, case_based_vectors)
            else:
                tag_scores = model(sentence_in)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    
    model = model.to(device)
    criterion = criterion.to(device)

# helper function for the train bilstm: 
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

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
    word_to_ix = model.word_to_ix
    sentence_lower = [word.lower() if word in word_to_ix.keys() else UNK for word in sentence]
    
    with torch.no_grad():
        inputs = prepare_sequence(sentence_lower, word_to_ix)
        tag_scores = model(inputs)
    
    tags_idx = np.argmax(tag_scores, axis = 1).tolist()
    
    idx_tag_dict = {v:k for k,v in model.tag_to_ix.items()}
    
    predicted_tags = [idx_tag_dict[i] for i in tags_idx]
    
    tagged_sentence = [(word, tag) for word, tag in zip(sentence, predicted_tags)]

    #TODO complete the code

    return tagged_sentence

def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    #TODO How do we know the input dimensions? 
    model_params = {'input_dimension': 16654,
                    'embedding_dimension': 100,
                    'num_of_layers': 2, #TODO add hidden layers
                    'output_dimension': 17}

    return model_params


##: Helper function to use pretrained embeddings
def create_weighted_matrix(target_vocab, vectors, emb_dim):
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0
    
#TODO make sure we can enumarate a dictionary
    for i, word in enumerate(target_vocab.keys()):
        try: 
            weights_matrix[i] = vectors[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    print(words_found)
    return weights_matrix

   

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
