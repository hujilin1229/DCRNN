from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import os
from lib.utils import load_graph_data1
from model.dcrnn_supervisor import DCRNNSupervisor

def main(args):

    config_filename = './data/model/dcrnn_{}.yaml'.format(args.city)
    with open(config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_mx = load_graph_data1(graph_pkl_filename)


        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': -1})
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
            print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
            supervisor.train(sess=sess)

        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
            supervisor.load(sess, supervisor_config['train']['model_filename'])
            outputs = supervisor.evaluate(sess)
            # np.savez_compressed(args.output_filename, **outputs)
            # print('Predictions saved as {}.'.format(args.output_filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--cuda', default='0', type=str, help='which gpu to use.')
    parser.add_argument('--city', default='Berlin', type=str, help='Which city to run exp.')
    args = parser.parse_args()
    main(args)
