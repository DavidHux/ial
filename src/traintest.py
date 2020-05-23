from itl import IALGE
from u import Preprocess, spa
from disturbEdges import distEdge_ran
from modelTrain import ialmodel_gcn
import numpy as np
import scipy.sparse as sp
from u import Eval
import warnings
import os
import random
import copy

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dst = ['cora', 'citeseer', 'pubmed']

dsindex = 0
distedges = True

dataset = dst[dsindex]

import pickle
def readpkl(name):
    # name = '/Users/davidhu/Desktop/npz/abcadj.pkl'
    with open(name, 'rb') as f:
        a = pickle.load(f)
    return a

def upper_triangular_mask(shape, as_array=False):
    a = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(i + 1, shape[1]):
            a[i, j] = 1.
    return a

def sample(probs, mask, pc=1.):
    # e = np.floor(np.random.random_sample(probs.shape) * mask * np.random.choice(2, probs.shape,  p=[1. - pc, pc]) + probs)
    e = np.floor(np.random.random_sample(probs.shape) * mask * pc + probs)
    res = sp.csr_matrix(e + e.T)
    res[res > 1] = 1
    return res

def top(probs, mask, to):
    s = sp.csr_matrix(probs*mask)
    rs = []
    t = s.nonzero()
    data = s.data
    rows = t[0]
    cols = t[1]
    for i in range(len(data)):
        rs.append((data[i], rows[i], cols[i]))
    ss = sorted(rs, key= lambda x: x[0], reverse=True)
    for i in range(0, 7000, 50):
        print(ss[i])
    dd = []
    rr = []
    cc = []
    for i in range(to):
        dd.append(1)
        rr.append(rs[i][1])
        cc.append(rs[i][2])
    csr = sp.csr_matrix((dd, (rr, cc)), shape=probs.shape)
    return csr + csr.T

def file_loader_generator(dir):
    dirs = os.listdir(dir)
    for d in dirs:
        if not d.endswith('.pkl'):
            continue
        pkl = readpkl(dir + d)
        yield pkl

pkldir = [
    '/Users/davidhu/Desktop/pkl/cora-25/',
    '/Users/davidhu/Desktop/pkl/cora-50/', 
    '/Users/davidhu/Desktop/pkl/cora-75/',
    '/Users/davidhu/Desktop/pkl/cora-100/',
    
]
ramainsize = [0.7554,0.9375,1.04657,1.102]
ramainsize_ci = [0.88,0.8921,0.6841,0.985]

if __name__ == "__main__":
    # adj, features, labels = Preprocess.loaddata('data/{}.npz'.format(dataset), llc=False)
    # _A_prev = _A_obs
    evaltimes = 20
    adj, features, labels = Preprocess.loaddata('data/{}-default.npz'.format(dataset))
    train, val, test = split_t = Preprocess.load_split('data/{}-default-split.pkl'.format(dataset))
    ial = ialmodel_gcn(adj, features, labels, split_t)
    for ii,ds in enumerate(pkldir):
        if ii == 0 or ii == 1:
            continue
        print('dir name {}'.format(ds))
        pkl_loader = file_loader_generator(ds)
        try:
            while True:
                adj_p = next(pkl_loader)
                mask = upper_triangular_mask(adj_p.shape)
                actual = delll = 0
                results = []
                for i in range(evaltimes):
                    # adj_sam = top(a, mask, 2714+500*9)
                    # adj_sam_del = sample(adj_p, mask, pc=ramainsize[ii])
                    adj_sam = sample(adj_p, mask)
                    # print(adj_sam.nnz / 2, adj_sam_del.nnz/2)
                    # actual += adj_sam.nnz / 2
                    # delll += adj_sam_del.nnz/2
                    # continue
                    embds = None
                    embs, pf = ial.singleTrain(adj_sam)
                    # te, va, tr = ial.multitest(adj_sam)
                    results.append(pf)
                    if embds is None:
                        embds = embs
                    else:
                        embds += embs
                embds /= evaltimes
                print(Eval.acu(embds[test], labels[test]))
                print('max {}, min {}, mean {}'.format(max(results), min(results), np.mean(results)))
        except StopIteration:
            # sys.exit()
            print('-----end-----')
    

    exit(0)
    a = readpkl()
    mask = upper_triangular_mask(a.shape)

    adj, features, labels = Preprocess.loaddata('data/{}-default.npz'.format(dataset))
    train, val, test = split_t = Preprocess.load_split('data/{}-default-split.pkl'.format(dataset))
    ial = ialmodel_gcn(adj, features, labels, split_t)
    te, va, tr = ial.multitest(adj)
    print(te, va, tr)
    exit(0)

    
    times = 2
    rrr = []
    for i in range(times):
        # adj_sam = top(a, mask, 2714+500*9)
        adj_sam = sample(a, mask, pc=0.8)
        # print(adj_sam.nnz)
        embds = None
        embs, pf = ial.singleTrain(adj_sam)
        te, va, tr = ial.multitest(adj_sam)
        # print(pf)
        if embds is None:
            embds = embs
        else:
            embds += embs
        rrr.append(te)
        print(te, va, tr)
    embds /= times
    print(Eval.acu(embds[test], labels[test]))
    
    print('max {}, min {}, mean {}'.format(max(rrr), min(rrr), np.mean(rrr)))