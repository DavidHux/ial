from u import loaddataset
from modelTrain import ialmodel_gcn
from u import Preprocess
import os
import warnings
warnings.filterwarnings("ignore")

def SGCtest(adj, ial):
    te, va, tr = ial.multitest(adj, t=1)
    return te


edgelistdir = '/Users/davidhu/Desktop/npz/coradeepwalkemd/'
percents = ['0.25', '0.5', '0.75', '1)']
datasets = ['cora', 'citeseer']
prf = ['init', 'final']
usesgc = True

def test():
    # dirs = os.listdir(edgelistdir)
    for dataset in datasets:
        _adj, feature, label = Preprocess.loaddata('data/{}-default.npz'.format(dataset), llc=False)
        split_t = Preprocess.load_split('data/{}-default-split.pkl'.format(dataset))
        ial = ialmodel_gcn(_adj, feature, label, split_t, sgc=usesgc)
        SGCtest(_adj, ial)
        # for percent in percents:
        #     for pr in prf:
        #         results = []
        #         sss = '{}-{}-{}'.format(dataset, percent, pr)
        #         print(sss)
        #         for d in dirs:
        #             if d.find(dataset) == -1 or d.find(percent) == -1 or d.find(pr) == -1:
        #                 continue
        #             try:
        #                 adj = loaddataset.load_edgelist2adj(edgelistdir+d, _adj.shape[0], deletei=1)
        #                 acu = SGCtest(adj, ial)
        #                 print(acu)
        #                 results.append(acu)
        #             except Exception as e:
        #                 print(e, d)
        #                 continue


if __name__ == "__main__":
    test()