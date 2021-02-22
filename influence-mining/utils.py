import numpy as np
import scipy.sparse as sp
import torch
import random
import pickle as pkl
import networkx as nx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def split_training_samples(idx_list):
    """Shuffle"""
    n = len(idx_list)
    random.shuffle(idx_list)
    end_train = int(n * 0.4)
    end_val = int(n * 0.8)

    """Split training samples"""
    idx_train = idx_list[:end_train - 1]
    idx_val = idx_list[end_train:end_val - 1]
    idx_test = idx_list[end_val:]

    return idx_train, idx_val, idx_test


def load_artist_data(path="../data"):
    """Load citation network dataset (default: cora)"""
    print("Loading artists dataset...")

    slice_idx = [1, 2, 4, 6, 7, 9, 10, 11]
    idx_features_labels = np.genfromtxt("{}/artists.prof".format(path), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, slice_idx], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/artists.conn".format(path), dtype=np.int32)
    edges = []
    for edge in edges_unordered:
        if edge[0] in idx_map and edge[1] in idx_map:
            edges.append([idx_map[edge[0]], idx_map[edge[1]]])
    edges = np.array(edges, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    num_features = features.shape[0]
    ran_idx = list(range(num_features))
    random.shuffle(ran_idx)

    num_train = int(num_features * 0.4)
    num_val = int(num_features * 0.7)
    idx_train = ran_idx[:num_train]
    idx_val = ran_idx[num_train: num_val]
    idx_test = ran_idx[num_val: num_features]

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, idx


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def parse_index_file(filename):
    """Parse index file (TensorFlow format)."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def to_tuple(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
