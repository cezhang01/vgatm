import argparse
import numpy as np
from model import Model
import os
from data_loader import Data


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--num_epochs', type=int, default=15)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=128)
    parser.add_argument('-dn', '--dataset_name', type=str, default='ml')
    parser.add_argument('-nt', '--num_topics', type=int, default=64)
    parser.add_argument('-sup', '--supervision', type=int, default=0)
    parser.add_argument('-reg_sup', '--reg_sup', type=float, default=1)
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8)
    parser.add_argument('-nn', '--num_sampled_neighbors', type=int, default=5)
    parser.add_argument('-ws', '--word_word_graph_window_size', type=int, default=5)
    parser.add_argument('-wn', '--word_word_graph_num_neighbors', type=int, default=5)
    parser.add_argument('-neg', '--num_negative_samples', type=int, default=5)
    parser.add_argument('-nl', '--num_convolutional_layers', type=int, default=2)
    parser.add_argument('-div', '--divergence', type=str, default='wasserstein')
    parser.add_argument('-p', '--prior', type=str, default='normal')
    parser.add_argument('-reg_div', '--reg_divergence', type=float, default=0.01)
    parser.add_argument('-reg_l2', '--reg_l2', type=float, default=1e-3)
    parser.add_argument('-kp', '--dropout_keep_prob', type=float, default=0.9)
    parser.add_argument('-ap', '--author_prediction', type=int, default=0)
    parser.add_argument('-rs', '--random_seed', type=int, default=520)
    parser.add_argument('-gpu', '--gpu', type=int, default=0)

    return parser.parse_args()


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.random_seed:
        np.random.seed(args.random_seed)
    data = Data(args)
    model = Model(args, data)
    print('Start training...')
    model.train()


if __name__ == '__main__':
    main(parse_arguments())