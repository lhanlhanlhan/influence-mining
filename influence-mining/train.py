from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_artist_data, accuracy
from models import AIMN, SpAIMN
import data as dt
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='Sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dt.data_preprocess()
adj, features, labels, idx_train, idx_val, idx_test, idx = load_artist_data()

# Model and optimizer
if args.sparse:
    model = SpAIMN(nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=int(labels.max()) + 1,
                  dropout=args.dropout,
                  nheads=args.nb_heads,
                  alpha=args.alpha)
else:
    model = AIMN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

"""训练完成，保存注意力系数"""
att_val = model.out_att.attention
att_val = torch.sum(att_val, dim=0)
att_val = att_val.cpu().detach().numpy()
att_val_dict = {}
with open("./att_value.txt", "w") as file:
    for i, val in enumerate(att_val):
        att_val_dict[idx[i]] = val
        file.write("{}\t{}\n".format(idx[i], val))

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

attr = ['danceability', 'energy', 'tempo', 'mode', 'key', 'instrumentalness', 'liveness', 'speechiness']

w_val = model.attentions[0].W
w_val = torch.sum(w_val, dim=-1)
w_val = w_val.cpu().detach().numpy()
print(w_val)
with open("./wei_value.txt", "w") as file:
    for i, val in enumerate(w_val):
        file.write("{}\t{}\n".format(attr[i], val))


# print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
#
# # Testing
# compute_test()
print("Analysing...")

"""分析重要程度"""
att_val = sorted(att_val_dict.items(), key=lambda x: x[1], reverse=True)
_, _, names = dt.read_influence()
head_artists = att_val[:40]
artist_influence_names = [names[i] for i, val in head_artists]
artist_influence_values = [val for _, val in head_artists]
with open("influence_sorted.csv", "w") as f:
    for i, val in head_artists:
        f.write("{},{}\n".format(names[i], val))

plt.switch_backend('agg')
# plt.figure(figsize=(12.0, 10.0))
# plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.bar(artist_influence_names, artist_influence_values)
plt.xticks(rotation=270)  # 横坐标竖着显示
plt.savefig("influence.pdf")
