# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import re
import pickle 
import csv
import time
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from torchtext import data
from torchtext.data import Field, TabularDataset, BucketIterator, Dataset
from torchtext.vocab import Vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


torch.manual_seed(1)


########### Remove before submission: 
from tqdm import tqdm



TRAIN_FILE_NAME = "trump_train.tsv"
TEST_FILE_NAME = "trump_test.tsv"
MODEL_SAVED_PATH = './model/m.pkl'
BEST_MODEL_PATH = './model/best_model.pkl'
TRAIN_MODEL = False
CREATE_EMBEDING = False

class RNN(nn.Module):
    def __init__(self, input_dim,output_dim, embedding_dim, hidden_dim):
        
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))

class LSTM(nn.Module):

    def __init__(self, embedding_model, dimension=5):
        super(LSTM, self).__init__()

        self.embeddings = embedding_model.embeddings
        self.word_to_ix = embedding_model.word_to_ix
        self.max_tweet_len = embedding_model.max_tweet_len
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size = self.embeddings.embedding_dim,
                            hidden_size=dimension,
                            num_layers=1)
        # self.drop = nn.Dropout(p=0.1)

        self.fc = nn.Linear(self.dimension, 1)
        self.act = nn.Sigmoid()

    def forward(self, tweet):
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
        text_emb = self.embeddings(lookup_tensor)

        h_0 = Variable(torch.zeros(1, 1, self.dimension))
        c_0 = Variable(torch.zeros(1, 1, self.dimension))
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(text_emb.view(len(lookup_tensor), 1, -1), (h_0, c_0))
        # lstm_out = lstm_out[-1:,:,:]
        final_hidden_state = lstm_out[-1]
        output_space = self.fc(final_hidden_state)
        output_scores = torch.sigmoid(output_space)
        return output_scores
    
    # def forward(self, tweet, hidden):


    #     # h_0 = Variable(torch.zeros(1, 1, self.dimension))
    #     # c_0 = Variable(torch.zeros(1, 1, self.dimension))
    #     lstm_out, hidden = self.lstm(text_emb.view(len(lookup_tensor), 1, -1), hidden)
    #     # lstm_out = lstm_out[-1:,:,:]
    #     drop_output = self.drop(lstm_out)
    #     drop_output = drop_output.contiguous().view(-1, self.dimension)

    #     # final_hidden_state = final_hidden_state[-1]
    #     output_space = self.fc(drop_output[-1])
    #     output_scores = torch.sigmoid(output_space)
    #     return output_scores, hidden


class NN(nn.Module):

    def __init__(self,input_size, output_size, layers,NN_epochs, p=0.1):
        super().__init__()
        
        self.NN_epochs = NN_epochs
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
    def train_model(self,X_train,y_train):
        losses = []
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        
        for epoch in range(self.NN_epochs):
            self.zero_grad()
            # total_loss = 0
           
            predctions = self(torch.tensor(X_train))
            
            loss = loss_function(predctions, torch.tensor(y_train, dtype=torch.float))
    
            loss.backward()
            optimizer.step()
        
            losses.append(loss.item())
        self.losses = losses
        
    def evaluate(self,X_test,y_test):
        with torch.no_grad():
            y_val = self(torch.tensor(X_test))
        y_val = np.argmax(y_val, axis=1)
        y_test = np.argmax(np.array(y_test), axis=1)
    
        metrics = classification_report(y_test,y_val,output_dict=True)
        metrics = pd.DataFrame(metrics).transpose()
        
        #: transform metrices to result format
        result = {}
        result["test_f1"] = metrics["f1-score"]["weighted avg"]
        result["test_accuracy"] = metrics["support"]["accuracy"]
        result['test_recall'] = metrics["recall"]["weighted avg"]
        return result
     

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
    # plt.hist(lengts,bins = 300)
    
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
    columns = ["user_handle", "tweet_text", "time_stamp"]
    #:reading data files
    test  = pd.read_csv(TEST_FILE_NAME,sep='\t', engine='python',names = columns,quoting=csv.QUOTE_NONE)  
    m.predict(test)
    
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

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train_func(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_recall = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)
            
            predictions = torch.round(torch.sigmoid(predictions))
            f1 = f1_score(batch.label, predictions, average='weighted')
            recall = recall_score(batch.label,predictions , average='weighted')

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1
            epoch_recall += recall
    
    
    result = {}
    result["test_f1"] =epoch_f1 / len(iterator)
    result["test_accuracy"] = epoch_acc / len(iterator)
    result['test_recall'] = epoch_recall / len(iterator)
    return epoch_loss / len(iterator),result

def preprocess_LSTM(train,batch_size,split_by = 0.8):
    tt = TweetTokenizer()
    TEXT = data.Field(tokenize = (lambda s: tt.tokenize(s)))
    LABEL = data.LabelField(dtype = torch.float)
    
    fields = [('text', TEXT), ('label', LABEL)]
    examples = []
    for i, row in train.iterrows():
        label = row["label"]
        text = row['tweet_text']
        examples.append(data.Example.fromlist([text, label], fields))
        
    train_dataset = Dataset(examples, fields)

    train_data, test_data = train_dataset.split(split_by)
    MAX_VOCAB_SIZE = 25000
    
    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data), 
        batch_size = batch_size,
        sort_key= lambda x:len(x.text), 
        sort_within_batch=False)
    return train_iterator,test_iterator

def train_LSTM(model,train_iterator,LSTM_params):
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(LSTM_params["N_EPOCHS"]):
         
        start_time = time.time()
        
        train_loss, train_acc = train_func(model, train_iterator, optimizer, criterion)
        
        end_time = time.time()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        
    return criterion


def run_ML_models(train,test,run_LR,run_linear_SVM,
                  run_SVM,run_NN,run_LSTM,run_boosting,
                  NN_params,LSTM_params):
    #: prepare x,y for basic models (LR,SVM,boosting)
    X = train["features"].to_list()
    y = train["label"].to_list()
    scoring = ['f1','accuracy', 'recall'] 
    
    ### Dimensionality reduction
    
    X_embedded = TSNE(n_components=2).fit_transform(X)
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=X_embedded[:,0], y=X_embedded[:,1],
    hue=y,
    palette=sns.color_palette("hls", 2),
    legend="full",
    alpha=0.3
)
    
    all_models_results = {}
    
    #: Logistic regression classifier
    if run_LR:
        LR_clf = LogisticRegression(random_state=0)
        cv_results_LR = cross_validate(LR_clf, X, y, cv=5, scoring = scoring)
        #:add to all models data
        all_models_results["LR"] = cv_results_LR
    
    
    #: SVC
    if run_linear_SVM:
        SVC_clf = SVC(gamma='auto',kernel = "linear")
        cv_results_SVC = cross_validate(SVC_clf, X, y, cv=5, scoring = scoring)
        all_models_results["SVM_linear"] = cv_results_SVC
        §
    if run_SVM:
        SVC_k_clf = SVC(gamma='auto')
        cv_results_SVC_k = cross_validate(SVC_k_clf, X, y, cv=5, scoring = scoring)
        all_models_results["SVM"] = cv_results_SVC_k
    
    #: GradientBoostingClassifier
    if run_boosting:
        GB_clf = GradientBoostingClassifier(random_state=0)
        cv_results_GB = cross_validate(GB_clf, X, y, cv=5, scoring = scoring)
        all_models_results["GB"] = cv_results_GB
    
    #: NN
    if run_NN:
        y_nn = [[1,0] if yi==0 else [0, 1] for yi in y ]
        X_train, X_test, y_train, y_test = train_test_split(X, y_nn, test_size=0.2, random_state=42)
        ##NN
        nn_clf = NN(len(X_train[0]),2, **NN_params)
        nn_clf.train_model(X_train,y_train)
        all_models_results["NN"]  = nn_clf.evaluate(X_test,y_test)

    if run_LSTM:
        ## LSTM
        train_iterator,test_iterator = preprocess_LSTM(train,LSTM_params["BATCH_SIZE"])
        
        INPUT_DIM = len(train_iterator.dataset.fields["text"].vocab)

        OUTPUT_DIM = 1
      
        model= RNN(INPUT_DIM,OUTPUT_DIM, LSTM_params["EMBEDDING_DIM"], LSTM_params["HIDDEN_DIM"])
        criterion = train_LSTM(model,train_iterator,LSTM_params)
        test_loss, result = evaluate(model, test_iterator, criterion)
        
        all_models_results["LSTM"] = result
            
    return all_models_results
    
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    embeddings_params = {"max_tweet_len" : 50,
                         "window_size":2, 
                         "embedding_size":10,
                         "embedding_neurons_1":128}
    
    general_params = { "epochs": 1,
                      "NN_epochs": 2500, 
                      "LSTM_epochs" :1}
    
    debug_params = {"run_LR": False,
                    "run_linear_SVM": False,
                    "run_SVM": False,
                    "run_boosting": False,
                    "run_NN": False,
                    "run_LSTM": False}
    
    NN_params = {"NN_epochs" : 100,
                 "layers" :[524,128],
                 "p":0.1}
    
    LSTM_params = {"NN_epochs" : 1,
                   "BATCH_SIZE" : 10,
                   "EMBEDDING_DIM"  :10,
                   "HIDDEN_DIM" : 124,
                   "N_EPOCHS": 1}
    
    deepNet_params = {"NN_params": NN_params,
                      "LSTM_params": LSTM_params}
    
    #: start preprocess over all the tweets in the train data
    train,test = preprocess()
    
    embedding_model = create_embedder(train, embeddings_params, general_params)
    if CREATE_EMBEDING:
        #:create  input vectors:
        create_embeding(train,embedding_model)
    else:
        with open('embedingData.pickle', 'rb') as handle:
            train = pickle.load(handle)

    #: run the all models no preprocessed data
    results = run_ML_models(train,test,**debug_params,**deepNet_params)
    
    for name, result in results.items():
        print(f" {name} \n f1: {np.mean(result['test_f1'])},acc: {np.mean(result['test_accuracy'])}, recall: {np.mean(result['test_recall'])}")
    
    
if __name__=='__main__':
    
    main() 

    
    

    

    