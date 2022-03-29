import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class IODataset(Dataset):
    """Create dataset from data.
    Parameters
    ----------
    u, y: ndarray, shape (total_len, n_channels) or (total_len,)
        Input and output signals. It should be either a 1d array or a 2d array.
    seq_len: int (optional)
        Maximum length for a batch on, respectively. If `seq_len` is smaller than the total
        data length, the data will be further divided in batches. If None,
        put the entire dataset on a single batch.
    """
    def __init__(self, u, y, seq_len=None):
        if seq_len is None:
            seq_len = u.shape[0]
        self.u = IODataset._batchify(u.astype(np.float32), seq_len)
        self.y = IODataset._batchify(y.astype(np.float32), seq_len)
        self.ntotbatch = self.u.shape[0]
        self.seq_len = self.u.shape[2]
        self.nu = 1 if u.ndim == 1 else u.shape[1]
        self.ny = 1 if y.ndim == 1 else y.shape[1]

    def __len__(self):
        return self.ntotbatch

    def __getitem__(self, idx):
        return self.u[idx, ...], self.y[idx, ...]

    @staticmethod
    def _batchify(x, seq_len):
        # data should be a torch tensor
        # data should have size (total number of samples) times (number of signals)
        # The output has size (number of batches) times (number of signals) times (batch size)
        # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
        nbatch = x.shape[0] // seq_len
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        x = x[:nbatch * seq_len]
        #    data = data[range(nbatch * batch_size), :]
        #    data = np.reshape(data, [batch_size, nbatch, -1], order='F')
        #    data = np.transpose(data, (1, 2, 0))
        # Evenly divide the data across the batch_size batches and make sure it is still in temporal order
        #    data = data.reshape((nbatch, 1, seq_len)).transpose(0, 1, 2)
        x = x.reshape((seq_len, nbatch, -1), order='F').transpose(1, 2, 0)
        # data = data.view(nbatch, batch_size, -1).transpose(0, 1)
        # ## arg = np.zeros([1, 2], dtype=np.float32)

        return x


def run_narendra_li_sim(u):
    # see andreas lindholm's work "A flexible state-space model for learning nonlinear dynamical systems"

    # get length of input
    k_max = u.shape[-1]

    # allocation
    x = np.zeros([2, k_max + 1])
    y = np.zeros([1, k_max])

    # run over all time steps
    for k in range(k_max):
        # state 1
        x[0, k + 1] = (x[0, k] / (1 + x[0, k] ** 2) + 1) * np.sin(x[1, k])
        # state 2
        term1 = x[1, k] * np.cos(x[1, k])
        term2 = x[0, k] * np.exp(-1 / 8 * (x[0, k] ** 2 + x[1, k] ** 2))
        term3 = u[0, k] ** 3 / (1 + u[0, k] ** 2 + 0.5 * np.cos(x[0, k] + x[1, k]))
        x[1, k + 1] = term1 + term2 + term3
        # output
        term1 = x[0, k] / (1 + 0.5 * np.sin(x[1, k]))
        term2 = x[1, k] / (1 + 0.5 * np.sin(x[0, k]))
        y[0, k] = term1 + term2

    return y


def create_narendra_li_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    # define output noise
    sigma_out = np.sqrt(0.1)

    # length of all data sets
    if bool(kwargs):
        k_max_train = kwargs['k_max_train']
        k_max_val = kwargs['k_max_val']
        k_max_test = kwargs['k_max_test']
    else:
        # Default option
        k_max_train = 50000
        k_max_val = 5000
        k_max_test = 5000

    # training / validation set input
    u_train = (np.random.rand(1, k_max_train) - 0.5) * 5
    u_val = (np.random.rand(1, k_max_val) - 0.5) * 5
    # test set input
    file_path = 'data/Narendra_Li/narendra_li_testdata.npz'
    test_data = np.load(file_path)
    u_test = test_data['u_test'][0:k_max_test]
    y_test = test_data['y_test'][0:k_max_test]

    # get the outputs
    y_train = run_narendra_li_sim(u_train) + sigma_out * np.random.randn(1, k_max_train)
    y_val = run_narendra_li_sim(u_val) + sigma_out * np.random.randn(1, k_max_val)

    # get correct dimensions

    u_train = u_train.transpose(1, 0)
    y_train = y_train.transpose(1, 0)
    u_val = u_val.transpose(1, 0)
    y_val = y_val.transpose(1, 0)

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test

# model_train, model_test调用

#     if dataset == 'nl':
#         dataset_train, dataset_valid, dataset_test = create_narendra_li_datasets(dataset_options.seq_len_train,
#                                                                                  dataset_options.seq_len_val,
#                                                                                  dataset_options.seq_len_test,
#                                                                                  **kwargs)
#         # Dataloader
#         loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
#         loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
#         loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
