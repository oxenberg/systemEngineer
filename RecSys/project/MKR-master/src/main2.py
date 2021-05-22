import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.02, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')

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
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--dim', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=2, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=1e-3, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')


show_loss = False
show_topk = False

music_args = {"dataset": "music",
              "n_epochs": 10,
              "dim" : 4,
              "L": 2,
              "H" : 1,
              "batch_size": 256,
              "l2_weight": 1e-6,
              "lr_rs": 1e-3,
              "lr_kge": 2e-4,
              "kge_interval": 2}

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)
