# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from nltk.tokenize import TweetTokenizer
import pickle 
import csv

torch.manual_seed(1)


########### Remove before submission: 
from tqdm import tqdm



TRAIN_FILE_NAME = "trump_train.tsv"
TEST_FILE_NAME = "trump_test.tsv"
MODEL_SAVED_PATH = './model/m.pkl'
TRAIN_MODEL = False
CREATE_EMBEDING = False




class NN(nn.Module):

    def __init__(self,input_size, output_size, layers, p=0.4):
        super().__init__()
     
        all_layers = []
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, inputs):
        x = self.layers(inputs)
        return x


class CBOW(nn.Module):

    def __init__(self, vocab,max_tweet_len, embedding_size, window_size, embedding_neurons_1):
        super(CBOW, self).__init__()
        self.word_to_ix = {word: i for i, word in enumerate(vocab, start=1)}
        self.vocab_size = len(vocab)+1    
        self.embeddings = nn.Embedding(self.vocab_size, embedding_size)
        self.linear1 = nn.Linear(2*window_size * embedding_size, embedding_neurons_1)
        self.linear2 = nn.Linear(embedding_neurons_1, self.vocab_size)
        self.max_tweet_len = max_tweet_len
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, inputs):
        # print(inputs)
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def predict(self,tweet):
        #create input vector
        #:inputs is a vector of words to idx (vector of 0,1)
        tt = TweetTokenizer()
        inputs = [self.word_to_ix[w] for w in tt.tokenize(tweet)]
        #padding
        inputs_len = len(inputs)
        add_zeros = self.max_tweet_len - inputs_len
        if add_zeros >= 0:
            inputs = [0]*add_zeros + inputs
        else:
            inputs = inputs[:self.max_tweet_len]
        lookup_tensor = torch.tensor(inputs, dtype=torch.long)
        embed = self.embeddings(lookup_tensor)
        return embed

def preprocess():
    
    trainColumns = ["tweet_id", "user_handle", "tweet_text", "time_stamp", "device"]
    testColumns = ["user_handle", "tweet_text", "time_stamp"]
    #:reading data files
    train  = pd.read_csv(TRAIN_FILE_NAME,sep='\t', engine='python',names = trainColumns,quoting=csv.QUOTE_NONE)    
    test  = pd.read_csv(TEST_FILE_NAME,sep='\t', engine='python',names = testColumns,quoting=csv.QUOTE_NONE)  
    
    # train["tweet_text"] = train["tweet_text"].apply(normalize_text)
    # test["tweet_text"] = test["tweet_text"].apply(normalize_text)
    
    #: Removing NaN
    train = train.dropna(axis = 0)
    test = test.dropna(axis = 0)
    #: Labeling the data based on the user handle and the device:
    #: if the user is realDonaldTrump and the device is android we assume it's Trump
    train["label"] = train.apply(lambda x :0 if x["user_handle"] == "realDonaldTrump" and x["device"] == "android" else 1  ,axis = 1)
    train.drop(["tweet_id", "device"], axis =1)
    return train,test

def create_vocab(data, window_size):
    tt = TweetTokenizer()
    tweets = [tt.tokenize(tweet) for tweet in data["tweet_text"].to_list()]
    
    #tweets size hist
    lengts = [len(tweet) for tweet in tweets]
    plt.hist(lengts,bins = 300)
    
    entire_text = [word for tweet in tweets for word in tweet]
    vocab = set(entire_text)

    cbow_data = []
    for tweet in tweets:
        for i in range(window_size, len(tweet) - window_size):
            context = tweet[i-window_size:i+window_size+1]
            del context[window_size]
        
            target = entire_text[i]
            cbow_data.append((context, target))
            
    return vocab, cbow_data

    
    
def load_best_model():
    """returning your best performing model that was saved as part of the submission bundle."""
    return None
def train_best_model():
    """training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
    Of course, the final model could be slightly different than the one returned by  load_best_model(), due to randomization issues.
    This function call training on the data file you received. 
    You could assume it is in the current directory. It should trigger the preprocessing and the whole pipeline. 
    """
    return None
def predict(m, fn):
    """ this function does expect parameters. m is the trained model and fn is the full path to a file in the same format as the test set (see above).
    predict(m, fn) returns a list of 0s and 1s, corresponding to the lines in the specified file.  
    """
    return None

def create_embedder(train_data, embeddings_params, general_params):
    vocab, cbow = create_vocab(train_data,embeddings_params["window_size"])
    #:find max tweet len:
    model = CBOW(vocab,**embeddings_params)

    if TRAIN_MODEL:    
    
        losses = []
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        
        # context_idxs = torch.tensor([model.word_to_ix[w] for w in cbow[35393][0]], dtype=torch.long)
        
        for epoch in range(general_params["epochs"]):
            total_loss = 0
            for context, target in tqdm(cbow):
        
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                context_idxs = torch.tensor([model.word_to_ix[w] for w in context], dtype=torch.long)
                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                model.zero_grad()
        
                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(context_idxs)
        
                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = loss_function(log_probs, torch.tensor([model.word_to_ix[target]], dtype=torch.long))
        
                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()
        
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            losses.append(total_loss)
        print(losses)
        torch.save(model.state_dict(), MODEL_SAVED_PATH)
    else:
        #: load model

        model.load_state_dict(torch.load(MODEL_SAVED_PATH))
        model.eval()
    
    return model

def create_embeding(data,embedding_model):
    flatten = lambda t: [item.tolist() for sublist in t for item in sublist]

    data["features"] = data["tweet_text"].apply(lambda x: flatten(embedding_model.predict(x)))
    with open('embedingData.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    embeddings_params = {"max_tweet_len" : 50,"window_size":2, 
              "embedding_size":10,
              "embedding_neurons_1":128}
    
    general_params = { "epochs": 1,
                      "NN_epochs": 100}
    
    train,test = preprocess()
    if CREATE_EMBEDING:
        embedding_model = create_embedder(train, embeddings_params, general_params)
        #:create  input vectors:
        create_embeding(train,embedding_model)
    else:
        with open('embedingData.pickle', 'rb') as handle:
            train = pickle.load(handle)
        
    
    X = train["features"].to_list()
    y = train["label"].to_list()
    
    ## Logistic regression classifier
    # LR_clf = LogisticRegression(random_state=0)
    # cv_results_LR = cross_validate(LR_clf, X, y, cv=5)
    # print(cv_results_LR['test_score'])


    # SVC
    # SVC_clf = SVC(gamma='auto',kernel = "linear")
    # cv_results_SVC = cross_validate(SVC_clf, X, y, cv=5)
    # print(cv_results_SVC['test_score'])

    # SVC_k_clf = SVC(gamma='auto')
    # cv_results_SVC_k = cross_validate(SVC_k_clf, X, y, cv=5)

    # print(cv_results_SVC_k['test_score'])
    
    ##NN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    layers = [128,128]
    nn_clf = NN(len(X_train[0]),2, layers)
    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(nn_clf.parameters(), lr=0.001)
    
    # context_idxs = torch.tensor([model.word_to_ix[w] for w in cbow[35393][0]], dtype=torch.long)
    
    for epoch in range(general_params["NN_epochs"]):
        total_loss = 0
        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        predctions = nn_clf(torch.tensor(X_train))

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(predctions, torch.tensor(y_train, dtype=torch.long))

        optimizer.zero_grad()


        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()
    
        # Get the Python number from a 1-element Tensor by calling tensor.item()
        losses.append(loss.item())


    with torch.no_grad():
        y_val = nn_clf(torch.tensor(X_test))
    y_val = np.argmax(y_val, axis=1)

    print(classification_report(y_test,y_val))


if __name__=='__main__':
    
    train = main()    

    