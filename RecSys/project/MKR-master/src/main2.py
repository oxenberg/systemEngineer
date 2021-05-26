import argparse
import numpy as np
from data_loader import load_data
from train import train
import time
from sklearn.model_selection import ParameterGrid

np.random.seed(555)

params = {"dim": [4, 8, 16],
          "L": [1, 2],
          "H": [1, 2],
          "batch_size": [16, 32, 128, 256, 512, 1024, 2048],
          "l2_weight": [1e-6, 1e-5, 1e-7],
          "lr_rs": [1e-3, 2e-4, 0.02, 0.0009],
          "lr_kge": [2e-4, 2e-5],
          "kge_interval": [1, 2],
          "conv_layer_filters": [4, 8, 16, 28],
          "dense_layer_filters": [4, 8, 16]
          }
param_grid = ParameterGrid(params)

parser = argparse.ArgumentParser()

'''
for dict_ in param_grid:
    
    # movie
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--dim', type=int, default=dict_["dim"], help='dimension of user and entity embeddings')
    parser.add_argument('--L', type=int, default=dict_["L"], help='number of low layers')
    parser.add_argument('--H', type=int, default=dict_["H"], help='number of high layers')
    parser.add_argument('--batch_size', type=int, default=dict_["batch_size"], help='batch size')
    parser.add_argument('--l2_weight', type=float, default=default=dict_["l2_weight"], help='weight of l2 regularization')
    parser.add_argument('--lr_rs', type=float, default=default=dict_["lr_rs"], help='learning rate of RS task')
    parser.add_argument('--lr_kge', type=float, default=default=dict_["lr_kge"], help='learning rate of KGE task')
    parser.add_argument('--kge_interval', type=int, default=default=dict_["kge_interval"], help='training interval of KGE task')
    parser.add_argument('--conv_layer_filters', type=int, default=dict_["conv_layer_filters"], help='convolutional layer number of filters')
    parser.add_argument('--dense_layer_filters', type=int, default=dict_["dense_layer_filters"], help='dense layer number of filters')
    '''
'''
for dict_ in param_grid:

    # book
    parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
    parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
    parser.add_argument('--dim', type=int, default=dict_["dim"], help='dimension of user and entity embeddings')
    parser.add_argument('--L', type=int, default=dict_["L"], help='number of low layers')
    parser.add_argument('--H', type=int, default=dict_["H"], help='number of high layers')
    parser.add_argument('--batch_size', type=int, default=dict_["batch_size"], help='batch size')
    parser.add_argument('--l2_weight', type=float, default=default=dict_["l2_weight"], help='weight of l2 regularization')
    parser.add_argument('--lr_rs', type=float, default=default=dict_["lr_rs"], help='learning rate of RS task')
    parser.add_argument('--lr_kge', type=float, default=default=dict_["lr_kge"], help='learning rate of KGE task')
    parser.add_argument('--kge_interval', type=int, default=default=dict_["kge_interval"], help='training interval of KGE task')
    parser.add_argument('--conv_layer_filters', type=int, default=dict_["conv_layer_filters"], help='convolutional layer number of filters')
    parser.add_argument('--dense_layer_filters', type=int, default=dict_["dense_layer_filters"], help='dense layer number of filters')
    '''

'''
for dict_ in param_grid:

    # music
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--dim', type=int, default=dict_["dim"], help='dimension of user and entity embeddings')
    parser.add_argument('--L', type=int, default=dict_["L"], help='number of low layers')
    parser.add_argument('--H', type=int, default=dict_["H"], help='number of high layers')
    parser.add_argument('--batch_size', type=int, default=dict_["batch_size"], help='batch size')
    parser.add_argument('--l2_weight', type=float, default=default=dict_["l2_weight"], help='weight of l2 regularization')
    parser.add_argument('--lr_rs', type=float, default=default=dict_["lr_rs"], help='learning rate of RS task')
    parser.add_argument('--lr_kge', type=float, default=default=dict_["lr_kge"], help='learning rate of KGE task')
    parser.add_argument('--kge_interval', type=int, default=default=dict_["kge_interval"], help='training interval of KGE task')
    parser.add_argument('--conv_layer_filters', type=int, default=dict_["conv_layer_filters"], help='convolutional layer number of filters')
    parser.add_argument('--dense_layer_filters', type=int, default=dict_["dense_layer_filters"], help='dense layer number of filters')
    '''

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.0009, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
parser.add_argument('--conv_layer_filters', type=int, default=28, help='convolutional layer number of filters')
parser.add_argument('--dense_layer_filters', type=int, default=16, help='dense layer number of filters')
'''
'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=2e-4, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-5, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
parser.add_argument('--conv_layer_filters', type=int, default=28, help='convolutional layer number of filters')
parser.add_argument('--dense_layer_filters', type=int, default=16, help='dense layer number of filters')
'''


# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')#10
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')#4
parser.add_argument('--L', type=int, default=2, help='number of low layers')#2
parser.add_argument('--H', type=int, default=1, help='number of high layers')#1
parser.add_argument('--batch_size', type=int, default=256, help='batch size')#256
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')#1e-6
parser.add_argument('--lr_rs', type=float, default=0.0009, help='learning rate of RS task') #1e-3
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
parser.add_argument('--conv_layer_filters', type=int, default=28, help='convolutional layer number of filters')
parser.add_argument('--dense_layer_filters', type=int, default=16, help='dense layer number of filters')

show_loss = False
show_topk = False

args = parser.parse_args()
data = load_data(args)
start_time = time.time()
train(args, data, show_loss, show_topk)
print("--- %s seconds to train and predict model ---" % (time.time() - start_time))

