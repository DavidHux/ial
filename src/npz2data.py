
from u import Preprocess
import numpy as np
import os

dataset = 'citeseer'
outfile = 'out'

ff = '/Users/davidhu/Desktop/res_citeseer/ial_res_{}{}.npz'
of = '/Users/davidhu/Desktop/res_citeseer/'
ff = ''
of = ''

suffix = ['_initadj', '_finaladj']

def savedata(adj, feature, label, outfilename):
    edgefile = of + '{}.edgelist'.format(outfilename)
    featurefile = of + '{}.feature'.format(outfilename)
    labelfile = of + '{}.label'.format(outfilename)
    
    with open(edgefile, 'w') as ef, open(featurefile, 'w') as ff, open(labelfile, 'w') as lf:
        farray = feature.toarray()
        print('len farray: {}'.format(len(farray)))
        for i in range(len(farray)):
            ff.write('{} {}\n'.format(i, ' '.join(map(str, farray[i]))))
            lf.write('{} {}\n'.format(i, label[i]))
        
        t = adj.nonzero()
        data = adj.data
        rows = t[0]
        cols = t[1]
        for i in range(len(data)):
            if rows[i] > cols[i]:
                continue
            ef.write('{} {}\n'.format(rows[i], cols[i]))


dirprefix = '/Users/davidhu/Desktop/npz/cora-default-0.25-0.5.npz/'
percents = ['0.25', '0.5', '0.75', '1)']

indir = '/root/hux/npz/'
# indir = dirprefix
outdir = '/root/hux/data/'

dataset = 'cora'

if __name__ == "__main__":

    dirs = os.listdir(indir)
    for p in percents:
        count = 0
        for d in dirs:
            print(d)
            a = d.find(p)
            if a == -1:
                continue

            ff = outdir+dataset
            if d.find('init') != -1:
                ff += 'init-{}+{}'.format(p, count)
            else:
                ff += 'final-{}+{}'.format(p, count)
            print(ff)
            count += 1
            adj, feature, label = Preprocess.loaddata(indir+d)
            adj = adj.tolil()
            for j in range(feature.shape[0]):
                if not adj[j].nonzero():
                    adj[j, j] = 1
            savedata(adj.tocsr(), feature, label, ff)
            