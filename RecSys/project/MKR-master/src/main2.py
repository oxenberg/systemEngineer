import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()


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
'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=2e-4, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-5, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
'''

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
'''

show_loss = False
show_topk = False
#
# music_args = {"dataset": "music",
#               "n_epochs": 10,
#               "dim" : 4,
#               "L": 2,
#               "H" : 1,
#               "batch_size": 256,
#               "l2_weight": 1e-6,
#               "lr_rs": 1e-3,
#               "lr_kge": 2e-4,
#               "kge_interval": 2}

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)
