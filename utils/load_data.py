from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy import sparse
import os

import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def load_netflix(args, **kwargs):

    unique_sid = list()
    with open(os.path.join(args.root_folder, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te


    # train, validation and test data
    x_train = load_train_data(os.path.join("datasets", "Netflix", 'train.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "Netflix", 'validation_tr.csv'),
                                         os.path.join("datasets", "Netflix", 'validation_te.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "Netflix", 'test_tr.csv'),
                                           os.path.join("datasets", "Netflix", 'test_te.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
# ======================================================================================================================
def load_dataset(args, **kwargs):
    if args.dataset_name == 'netflix':
        train_loader, val_loader, test_loader, args = load_netflix(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args
