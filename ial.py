
# 
# 
# 
# 
# 
# 
# 
# 
# a

import warnings
import os
import random
import copy

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.utils.u import Preprocess
from src.itl import IALGE
# from src.utils.util import checkargs


def start(args):
    '''
    start a network structure learn job
    args:
      dataset
      remainpercent
      ...

    '''
    print('start network structure learn')
    _adj, feas, labels = Preprocess.load_data('data/{}-default.npz'.format(args.dataset), llc=False)
    adj_d, remained, deleted = Preprocess.delete_edges(_adj, k=0.1)
    # adj = Preprocess.adj_add(adj_d, len(deleted))
        
    t = IALGE(adj_d, feas, labels, 100, 1000, 10, cangen='ran', edgeEval='SGC', edgeUpdate='topK',
                    seed=123, poolnum=20)

    t.run(recover=True)

def check(args):
    '''
    check the status of a struture learn job
    '''
    pass

def analyz(args):
    '''
    analyz a job
    '''
    pass



def main(args):
    if args.command == 'start':
        start(args)
    elif args.command == 'check':
        check(args)
    elif args.command == 'analyz':
        analyz(args)
    else:
        pass

if __name__ == '__main__':
    import argparse
    from src.const import configs

    parser = argparse.ArgumentParser(description='network structure learn tool')
    parser.add_argument('command', type=str)
    parser.add_argument('--name', default='temp', type=str,
                        help='The task name. Default: temp')
    parser.add_argument('--dataset', default='cora', type=str, choices=configs['datasets'],
                        help='The dataset: cora, citeseer, pubmed ' +
                        'Default: cora')
    parser.add_argument('--remain', default=50, type=int, choices=configs['remainsize'],
                        help='Remain edges percent. Default: 50')
    parser.add_argument('--dist', default='false', type=str, choices=configs['ifdist'],
                        help='Set whether disturbing the network structure. Accepted input: true, false. Default: false')
    parser.add_argument('--distsize', default=10, type=int, choices=configs['distsize'],
                        help='Disturn size. Default: 10.')
    parser.add_argument('--disttype', default='random', type=str, choices=configs['disttype'],
                        help='Disturb type. Default: random')

    parser.add_argument('--trainround', default=100, type=int,
                        help='Max train round. Default: 100')
    parser.add_argument('--minedgesnum', default=1000, type=int,
                        help='Min edges num to add. Default: 1000')
    parser.add_argument('--maxedgesnum', default=10000, type=int,
                        help='Max edges num to add. Default: 10000')
    parser.add_argument('--addnumeachround', default=10, type=int,
                        help='Size to add each round. Default: 10')
    parser.add_argument('--cangen', default='random', type=str, choices=configs['cangen'],
                        help='Cangen type. Default: random')
    parser.add_argument('--edgeeval', default='sgc', type=str, choices=configs['edgeeval'],
                        help='Edge eval type. Default: sgc')

    args = parser.parse_args()
    print(args)

    # _data, _method, _seed, _missing_percentage = args.d, args.m, args.s, args.e/100

    # main(_data, _method, _seed, _missing_percentage)
    main(args)
