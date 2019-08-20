import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf

from scipy.sparse import linalg
import h5py
import datetime
import re

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data

def load_dataset1(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['training', 'validation']:
        cat_data = np.load(os.path.join(dataset_dir, category + '_raw_data.npy'))
        data['x_' + category] = np.transpose(cat_data[:, :3, :, :], [0, 1, 3, 2]) / 255.
        data['y_' + category] = np.transpose(cat_data[:, 3:, :, :], [0, 1, 3, 2]) / 255.
    # scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    #     data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_training'], data['y_training'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_validation'], data['y_validation'], test_batch_size, shuffle=False)
    # data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = None

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_graph_data1(pkl_filename):
    adj_mx = sp.load_npz(pkl_filename)
    adj_mx = adj_mx.todense()

    return adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def create_directory_structure(root):
    berlin = os.path.join(root, "Berlin","Berlin_test")
    istanbul = os.path.join(root, "Istanbul","Istanbul_test")
    moscow = os.path.join(root, "Moscow", "Moscow_test")
    try:
        os.makedirs(berlin)
        os.makedirs(istanbul)
        os.makedirs(moscow)
    except OSError:
        print("failed to create directory structure")
        # sys.exit(2)

def load_input_data(file_path, indicies):
    """
    Given a file path, load the relevant training data pieces into a tensor that is returned.
    Return: tensor of shape (number_of_test_cases_per_file =5, 3, 495, 436, 3)
    """
    # load h5 file into memory.
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])


    # get relevant training data pieces
    data = [data[y-3:y] for y in indicies]
    data = np.stack(data, axis=0)

    # type casting
    data = data.astype(np.float32)
    return data

def list_filenames(directory, excluded_dates=[]):
    """Auxilliary function which returns list of file names in directory in random order,
        filtered by excluded dates.

        Args.:
            directory (str): path to directory
            excluded_dates (list): list of dates which should not be included in result list,
                e.g., ['2018-01-01', '2018-12-31']

        Returns: list
    """
    filenames = os.listdir(directory)
    # np.random.shuffle(filenames)

    if len(excluded_dates) > 0:
        # check if in excluded dates
        excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates]
        filenames = [x for x in filenames if return_date(x) not in excluded_dates]

    return filenames

def cast_moving_avg(data):
    """
    Returns cast moving average (cast to np.uint8)
    data = tensor of shape (5, 3, 495, 436, 3) of  type float32
    Return: tensor of shape (5, 3, 495, 436, 3) of type uint8
    """

    prediction = []
    for i in range(3):
        data_slice = data[:, i:]
        t = np.mean(data_slice, axis = 1)
        t = np.expand_dims(t, axis = 1)
        prediction.append(t)
        data = np.concatenate([data, t], axis =1)

    prediction = np.concatenate(prediction, axis = 1)
    prediction = np.around(prediction)
    prediction = prediction.astype(np.uint8)

    return prediction

def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def return_date(file_name):
    """Auxilliary function which returns datetime object from Traffic4Cast filename.

        Args.:
            file_name (str): file name, e.g., '20180516_100m_bins.h5'

        Returns: date string, e.g., '2018-05-16'
    """

    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date