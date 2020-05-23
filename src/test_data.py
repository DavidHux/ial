from itl import IALGE
from u import Preprocess, spa
from disturbEdges import distEdge_ran

import warnings
import os
import random
import copy

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dst = ['cora', 'citeseer', 'pubmed']

dsindex = 0
distedges = False
default = True
percent = [0.8]
pooln = 2
testnum = 6

stype = 'node'
dataset = dst[dsindex]

randomedgenum=10000 if dataset != 'pubmed' else 100000
edgenumPit2adds = 10 if dataset != 'pubmed' else 100
cannumpits = randomedgenum
knns = 10
subsetevalnum = 300
etimeperedge = 10

minedgesnum = 10000 if dataset == 'pubmed' else 2000


if dataset == 'cora':
    share = (0.052, 0.1847)
elif dataset == 'citeseer':
    share = (0.0362, 0.1504)
elif dataset == 'pubmed':
    share = (0.003, 0.1)
elif dataset == 'polblogs':
    share = (0.1, 0.4)


itn = int(minedgesnum / edgenumPit2adds * 2)


if __name__ == "__main__":
    if default:
        _A_obs, feas, labels = Preprocess.loaddata('data/{}-default.npz'.format(dataset), llc=False)
        # print(type(labels), labels.shape)
        # exit(0)
    else:
        _A_obs, feas, labels = Preprocess.loaddata('data/{}.npz'.format(dataset), llc=False)
    _A_prev = _A_obs
    
    for p in percent:
        ds = (dataset, p)
        if dataset == 'pubmed':
            split_seed = 36
        elif default == True:
            split_seed = -2
        else:
            split_seed = random.randint(1, 100)
        etimeperedge = split_seed
        pp = (edgenumPit2adds, cannumpits, knns, subsetevalnum, etimeperedge)
        for i in range(testnum):
            adj_d, remained, deleted = spa.delete_edges(_A_obs, k=p)
            print('preprocess, delete some edges, remaind edges num(bi): {}, prev: {}'.format(adj_d.nnz, _A_prev.nnz))
            if distedges == True:
                dist = distEdge_ran(len(deleted))
                print('dist edges, add {}'.format(len(deleted)))
                adj = dist.disturb(adj_d)
            else:
                adj = adj_d
            
            t = IALGE(adj, feas, labels, itn, minedgesnum, 10, cangen='ran', edgeEval='SGC', edgeUpdate='topK',
                    seed=i+1, dropout=0, randomedgenum=randomedgenum,
                    deleted_edges=deleted, disturbadj_before=adj_d, completeadj=_A_prev, params=pp, dataset=ds, split_share=share, simtype=stype, split_seed=split_seed,
                    poolnum=pooln)
            
            print('''\nstart...\n''')
            t.run(recover=distedges)
