import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data1
from model.dcrnn_supervisor import DCRNNSupervisor

utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174,222, 258]

def run_dcrnn(args):
    config_filename = './data/model/dcrnn_{}.yaml'.format(args.city)
    with open(config_filename) as f:
        config = yaml.load(f)
        graph_pkl_filename = config['data'].get('graph_pkl_filename')
        adj_mx = load_graph_data1(graph_pkl_filename)
        node_pos_pkl_filename = config['data'].get('node_pos_pkl_filename')
        node_pos = np.load(node_pos_pkl_filename)
        indicies = utcPlus3
        if args.city == 'Berlin':
            indicies = utcPlus2

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': -1})
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
            supervisor.load(sess, args.model_file)
            supervisor.pred_write_submission_files_with_avg(sess, args, node_pos, indicies)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--cuda', default='0', type=str, help='which gpu to use.')
    parser.add_argument('--city', default='Berlin', type=str, help='Which city to run exp.')
    parser.add_argument('--model_file', default=None, type=str, help="Saved model file")
    parser.add_argument('--data_dir', type=str, default='../data/', help='directory to load the data')
    parser.add_argument('--output_dir', type=str, default='../Results/GCN_TF/', help='directory to load the data')
    args = parser.parse_args()
    run_dcrnn(args)
