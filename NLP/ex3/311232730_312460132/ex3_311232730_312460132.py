# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import pickle 
import csv
import time
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import joblib
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

## If you have problems loading the pickle files, we used python 3.8.5

TRAIN_FILE_NAME = "trump_train.tsv" # If file names are different than states here, please change it
TEST_FILE_NAME = "trump_test.tsv" # If file names are different than states here, please change it
MODEL_SAVED_PATH = './model/embeddings.pkl'
BEST_MODEL_PATH = './model/best_model.pkl'
TRAIN_MODEL = False ##### Train Embedding Model?
CREATE_EMBEDING = True ##### Create Embedding on train data

class RNN(nn.Module):
    """
        RNN Class for the RNN model
    """
    def __init__(self, input_dim,output_dim, embedding_dim, hidden_dim):
        
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):

        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))



class NN(nn.Module):
    """
        NN Class for the NN Network
    """

    def __init__(self,input_size, output_size, layers,NN_epochs, p=0.1):
        """
            initiliaze the neurol network 
        """
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
        """
            training the network
        """
        losses = []
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        
        for epoch in range(self.NN_epochs):
            self.zero_grad()
           
            predctions = self(torch.tensor(X_train))
            
            loss = loss_function(predctions, torch.tensor(y_train, dtype=torch.float))
    
            loss.backward()
            optimizer.step()
        
            losses.append(loss.item())
        self.losses = losses
        
    def evaluate(self,X_test,y_test):
        """
           evaluating the network based on the metrics we defined for all models.  
        """
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
    """
     CBOW class helps us create and calculate the embedding vectors we use for all the models. 
    """

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
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def predict(self,tweet):
        #:create input vector
        #:inputs is a vector of words to idx (vector of 0,1)
        tt = TweetTokenizer()
        inputs = [self.word_to_ix[w] if w in self.word_to_ix.keys() else 0 for w in tt.tokenize(tweet)]
        #:padding
        inputs_len = len(inputs)
        add_zeros = self.max_tweet_len - inputs_len
        if add_zeros >= 0:
            inputs = [0]*add_zeros + inputs
        else:
            inputs = inputs[:self.max_tweet_len]
        lookup_tensor = torch.tensor(inputs, dtype=torch.long)
        embed = self.embeddings(lookup_tensor)
        return embed

def preprocess(test = True):
    """
        Preprocessing the data. The flag helps us understand if we use the test data or not. 
        The function reads the data files,remove rows with null values and for the train data
        adds the label. 
    """
    
    trainColumns = ["tweet_id", "user_handle", "tweet_text", "time_stamp", "device"]
    
    #:reading data files
    train  = pd.read_csv(TRAIN_FILE_NAME,sep='\t', engine='python',names = trainColumns,quoting=csv.QUOTE_NONE)    
    
    test = None
    if test: 
        testColumns = ["user_handle", "tweet_text", "time_stamp"]
        test  = pd.read_csv(TEST_FILE_NAME,sep='\t', engine='python',names = testColumns,quoting=csv.QUOTE_NONE)  
        test = test.dropna(axis = 0)
    
    #: Removing NaN
    train = train.dropna(axis = 0)
    #: Labeling the data based on the user handle and the device:
    #: if the user is realDonaldTrump and the device is android we assume it's Trump
    train["label"] = train.apply(lambda x :0 if x["user_handle"] == "realDonaldTrump" and x["device"] == "android" else 1  ,axis = 1)
    train.drop(["tweet_id", "device"], axis =1)
    return train,test

def create_vocab(data, window_size):
    tt = TweetTokenizer()
    tweets = [tt.tokenize(tweet) for tweet in data["tweet_text"].to_list()]
    
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
    
def create_embedder(train_data, embeddings_params, general_params, load_all_model = True):
    """
        If no embeddings exists the function trains the embedding model and saves the results to file. 
        Otherwise, it loads the trained embedding from file. 
    """ 

    if TRAIN_MODEL:    
    
        vocab, cbow = create_vocab(train_data,embeddings_params["window_size"])
        model = CBOW(vocab,**embeddings_params)
        losses = []
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        
        # context_idxs = torch.tensor([model.word_to_ix[w] for w in cbow[35393][0]], dtype=torch.long)
        
        for epoch in range(general_params["epochs"]):
            total_loss = 0
            for context, target in cbow:
        
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
        # torch.save(model.state_dict(), MODEL_SAVED_PATH)
        torch.save(model, MODEL_SAVED_PATH)
    else:
        #: load model
        model = torch.load(MODEL_SAVED_PATH)
        # model.load_state_dict(torch.load('./model/m.pkl'))
        model.eval()

    return model

def create_embeding(data,embedding_model):
    """
        The function uses the embedding vectors to create the feature for the data. 
        It predicts the embedding for each tweet in the data and adds the predicted embeddings 
        as features. 
    """
    flatten = lambda t: [item.tolist() for sublist in t for item in sublist]

    data["features"] = data["tweet_text"].apply(lambda x: flatten(embedding_model.predict(x)))

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    The function is used in the RNN model
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train_func(model, iterator, optimizer, criterion):
    """
       Train function for the RNN model 
    """

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
    """
       Evaluation function for the RNN model 
    """

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

def epoch_time(start_time, end_time):
    """
       The function calculates the time it takes for each epoch in the RNN
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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
    """
      Train RNN model 
    """
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

def apply_random_search(classifier, dist, n_iter, scoring, X, y, save = False): 
    """
       The function gets the classifier and the distribution we want to use in the randomized
       search, and returns the result for the best estimator and the best paramaters chosen by the 
       randomized search
    """
    randomized = RandomizedSearchCV(classifier, dist, random_state=0, cv = 5, scoring = scoring,
                                               n_iter = n_iter, refit = "accuracy", verbose = 1)
    search = randomized.fit(X, y)
    cv_results = search.cv_results_
    results = {}
    for scorer in scoring: 
        results['test_%s' %scorer] =np.nanmax(cv_results['mean_test_%s' %scorer])
    
    if save:
        with open(BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(search.best_estimator_, f)

    return results, search.best_params_
    
    

def run_ML_models(train,test,general_params ,run_LR,run_linear_SVM,
                  run_SVM,run_NN,run_LSTM,run_boosting,
                  NN_params,LSTM_params):
    """
       The function runs all 5 algorithms on demand. The user can choose which 
       of the algorithms he wants to run (if any). 
       It splits the data to X (features) and y (label), creates X_embedded with dimensionality
       reduction and uses tha data to train all models. 
    """
    
    
    #: prepare x,y for basic models (LR,SVM,boosting)
    X = train["features"].to_list()
    y = train["label"].to_list()
    scoring = ['f1','accuracy', 'recall'] 
    
    ### Dimensionality reduction
    
    X_embedded = PCA(n_components=100).fit_transform(X)
    
    all_models_results = {}
    best_params_for_all = {}
    
    #: Logistic regression classifier
    if run_LR:
        distributions_LR = {"C":np.arange(0, 4, 0.1), 
                        "max_iter": np.arange(600, 1000, 100)}
        LR_clf = LogisticRegression(random_state=0)
        #:add to all models data
        results, best_params = apply_random_search(LR_clf, distributions_LR,
                                                   general_params['n_iter'], scoring, X, y)
        all_models_results["LR"] = results
        best_params_for_all["LR"] = best_params
    
    
    #: SVC
    if run_linear_SVM:
        distributions_SVC = {"C":np.arange(0, 2, 0.1)}
        SVC_clf = SVC(gamma='auto',kernel = "linear")
        results, best_params = apply_random_search(SVC_clf, distributions_SVC,
                                                   general_params['n_iter'], scoring, X_embedded, y)
        all_models_results["SVM_linear"] = results
        best_params_for_all["SVM_linear"] = best_params
        
    if run_SVM:
        distributions_SVC_k = {"C":np.arange(0, 2, 0.1)}
        SVC_k_clf = SVC(gamma='auto')
        results, best_params = apply_random_search(SVC_k_clf, distributions_SVC_k,
                                                   general_params['n_iter'], scoring, X_embedded, y)
        all_models_results["SVM"] = results
        best_params_for_all["SVM"] = best_params
    
    #: GradientBoostingClassifier
    if run_boosting:
        distributions_GB = {"learning_rate":np.arange(0.01, 0.2, 0.01), 
                            "n_estimators": np.arange(100,500,100), 
                            "max_depth": np.arange(2,10,1)}
        GB_clf = GradientBoostingClassifier(random_state=0)
        results, best_params = apply_random_search(GB_clf, distributions_GB,
                                                   general_params['n_iter'], scoring, X, y, True)
        all_models_results["GB"] = results
        best_params_for_all["GB"] = best_params
    
    
    #: NN
    if run_NN:
        paramaters_grid = {"depth": np.arange(2,6,1),  
                           "number_of_neurons": np.arange(100, 1000,100)}
        params = list(ParameterGrid(paramaters_grid))
        shuffle(params)
        y_nn = [[1,0] if yi==0 else [0, 1] for yi in y]
        X_train, X_test, y_train, y_test = train_test_split(X, y_nn, test_size=0.2, random_state=42)
        ##NN
        accuracy = 0
        for param in params[:general_params['n_iter']]: 
            layers = param['depth']*[param['number_of_neurons']]
            NN_params["layers"] = layers
            nn_clf = NN(len(X_train[0]),2, **NN_params)
            nn_clf.train_model(X_train,y_train)
            results = nn_clf.evaluate(X_test,y_test)
            if results['test_accuracy'] > accuracy:
                all_models_results["NN"]  = results
                accuracy = results['test_accuracy']
                best_params_for_all["NN"] = param

    if run_LSTM:
        ## LSTM
        train_iterator,test_iterator = preprocess_LSTM(train,LSTM_params["BATCH_SIZE"])
        
        INPUT_DIM = len(train_iterator.dataset.fields["text"].vocab)

        OUTPUT_DIM = 1
      
        model= RNN(INPUT_DIM,OUTPUT_DIM, LSTM_params["EMBEDDING_DIM"], LSTM_params["HIDDEN_DIM"])
        criterion = train_LSTM(model,train_iterator,LSTM_params)
        test_loss, result = evaluate(model, test_iterator, criterion)
        
        all_models_results["LSTM"] = result
            
    return all_models_results, best_params_for_all
    

def main():
    """
       The full pipeline with all the paramaters needed to run all the algorithms. 
    """
    embeddings_params = {"max_tweet_len" : 35,
                         "window_size":2, 
                         "embedding_size":10,
                         "embedding_neurons_1":128}
    
    general_params = { "epochs": 10, 
                      "n_iter" : 2}
    
    # If you want to run training on some of the models, change their flag to true. 
    debug_params = {"run_LR": False,
                    "run_linear_SVM": False,
                    "run_SVM": False,
                    "run_boosting": False,
                    "run_NN": False,
                    "run_LSTM": False}
    
    NN_params = {"NN_epochs" : 1000,
                 "p":0.1}
    
    LSTM_params = {"NN_epochs" : 100,
                   "BATCH_SIZE" : 10,
                   "EMBEDDING_DIM"  :10,
                   "HIDDEN_DIM" : 124,
                   "N_EPOCHS": 10}
    
    
    deepNet_params = {"NN_params": NN_params,
                      "LSTM_params": LSTM_params}
    
    #: start preprocess over all the tweets in the train data
    train,test = preprocess()
    
    embedding_model = create_embedder(train, embeddings_params, general_params)

    #:create  input vectors:
    create_embeding(train,embedding_model)

    #: run the all models no preprocessed data
    results, best_params = run_ML_models(train,test,general_params, **debug_params,**deepNet_params)
    
    for name, result in results.items():
        print(f" {name} \n f1: {np.mean(result['test_f1'])},acc: {np.mean(result['test_accuracy'])}, recall: {np.mean(result['test_recall'])}")
     
    return best_params

def load_best_model():
    """returning your best performing model that was saved as part of the submission bundle."""
    model = joblib.load(BEST_MODEL_PATH)
    
    return model

def train_best_model():
    """training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
    Of course, the final model could be slightly different than the one returned by  load_best_model(), due to randomization issues.
    This function call training on the data file you received. 
    You could assume it is in the current directory. It should trigger the preprocessing and the whole pipeline. 
    """
    embeddings_params = {"max_tweet_len" : 35,
                         "window_size":2, 
                         "embedding_size":10,
                         "embedding_neurons_1":128}
    
    general_params = { "epochs": 10, 
                      "n_iter" : 2}
    
    train, _ = preprocess(False)
    embedding_model = create_embedder(train, embeddings_params, general_params)

    #:create  input vectors:
    create_embeding(train,embedding_model)
    
    X = train["features"].to_list()
    y = train["label"].to_list()
    
    GB_clf = GradientBoostingClassifier(random_state=0, n_estimators=200, learning_rate=0.09, max_depth=7)
    GB_clf.fit(X, y)
    
    return GB_clf

def predict(m, fn):
    """ this function does expect parameters. m is the trained model and fn is the full path to a file in the same format as the test set (see above).
    predict(m, fn) returns a list of 0s and 1s, corresponding to the lines in the specified file.  
    """
    embeddings_params = {"max_tweet_len" : 35,
                         "window_size":2, 
                         "embedding_size":10,
                         "embedding_neurons_1":128}
    
    general_params = { "epochs": 10, 
                      "n_iter" : 2}
    
    columns = ["user_handle", "tweet_text", "time_stamp"]
    #:reading data files
    test  = pd.read_csv(fn ,sep='\t', engine='python',names = columns,quoting=csv.QUOTE_NONE)  
    embedding_model = create_embedder(test, embeddings_params, general_params)
    create_embeding(test,embedding_model)
    X = test["features"].to_list()

    predictions = list(m.predict(X))
    
    
    return predictions
    

###### If you want to run the entire pipeline with all the algorithms you should uncomment this    
# if __name__=='__main__':
#     main()

    

    
    

    

    