import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from . import utils_nete as utils
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split #数据集划分

import scipy.sparse as sp
import networkx as nx
import pickle as pkl
# from gemodels import gemodel_GCN



class Sim():
    def min_sq(vec):
        ''' 最小二乘法
        args:
            vec: N*D array, features

        returns:
            res: ordered pairs, sorted by sims
        '''
        res = []
        for i in range(vec.shape[0]):
            for j in range(i+1, vec.shape[0]):
                sim = sum(pow(vec[i]-vec[j], 2))
                res.append([sim, i, j])
            # print(i, sim)

        res = sorted(res)
        return res

    def min_sq(vec1, vec2):
        assert(len(vec1) == len(vec2))
        res = sum(pow(vec1-vec2, 2))
        return res

from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
class loaddataset():
    def loadsk(dataset, k=1):
        feature, label = loaddataset.loadfl(dataset)
        train, val, test = loaddataset.split_sk_data(dataset)
        print(len(train), len(val))

        # print(feature, label, train)
        fea = sp.csr_matrix(feature)
        adj = sp.csr_matrix(([], ([],[])), shape=(feature.shape[0], feature.shape[0]))
        # adj = kneighbors_graph(feature, k, metric='cosine')
        return adj, fea, label, train, val, test
    
    def load_k_neibor(feature, k):
        return kneighbors_graph(feature, k, metric='cosine')
    
    def loadfl(dataset):
        if dataset == 'wine':
            data = load_wine()
        elif dataset == 'cancer':
            data = load_breast_cancer()
        elif dataset == 'digits':
            data = load_digits()
        else:
            print('load dataeset error, no {}'.format(dataset))
        
        
        feature = data['data']
        label = data['target']
        # print(feature.shape)
        return feature, label
    
    def load_data_gcn_kipf(dataset, k = 0):
        feature, label = loaddataset.loadfl(dataset)
        train, val, test = sets = loaddataset.split_sk_data(dataset)
        train_mask, val_mask, test_mask = loaddataset.load_mask(sets, feature.shape[0])
        y_train, y_val, y_test = loaddataset.load_onehot_label(label, sets, (feature.shape[0], max(label)+1))
        if k == 0:
            adj = sp.csr_matrix(([], ([],[])), shape=(feature.shape[0], feature.shape[0]))
        else:
            adj = loaddataset.load_k_neibor(feature, k)
        return adj, sp.csr_matrix(feature), y_train, y_val, y_test, train_mask, val_mask, test_mask
    
    def load_onehot_label(labels, splits, shape):
        res = [np.zeros(shape)] * 3
        for i in range(3):
            for x in splits[i]:
                res[i][x][labels[x]] = 1
        return res
    
    def load_mask(sets, N):
        res = [[False] * N] * len(sets)
        for i in range(len(sets)):
            for x in sets[i]:
                res[i][x] = True
        return res
    
    def loadsplit_1(dataset):
        if dataset == 'wine':
            size = 178
            tr = 10
            va = 10
        elif dataset == 'cancer':
            size = 569
            tr = 10
            va = 20
        elif dataset == 'digits':
            size = 1797
            tr = 50
            va = 100
            # return list(range(0, tr)), list(range(tr, va+tr)), list(range(va+tr, size))
        train = list(range(0,size,int(size/tr)))
        val = list(set(range(1,size,int(size/va/2))) - set(train))
        test = list(set(range(size)) - set(train) - set(val))
        return (train, val, test)
    
    def split_sk_data(dataset):
        if dataset == 'wine':
            size = 178
            tr = 10
            va = 20
        elif dataset == 'cancer':
            size = 569
            tr = 10
            va = 20
        elif dataset == 'digits':
            size = 1797
            tr = 50
            va = 100
        
        feature, y = loaddataset.loadfl(dataset)

        # data_train,data_test,target_train,target_test = train_test_split(np.arange(size),y,test_size=0.8,random_state=125)
        train, test, y_train, y_test = train_test_split(np.arange(size), y, train_size=tr+va,test_size=size - tr - va,
                                                        stratify=y)
        train, val, y_train, y_val = train_test_split(train, y_train, train_size=tr,test_size=va,
                                                        stratify=y_train)
        return (train.tolist(), val.tolist(), test.tolist())
   
    def load_sk_data_splited(dataset):
        feature, label = loaddataset.loadfl(dataset)
        train, val, test = loaddataset.split_sk_data(dataset)
        train.extend(val)
        # print(train)
        return feature[train], feature[test], label[train], label[test]
    
    def load_edgelist2adj(filename, N, deletei=0):
        row = []
        col = []
        with open(filename) as infile:
            for line in infile:
                strs = line.strip('\n').split(' ')
                row.append(int(strs[0]) - deletei)
                col.append(int(strs[1]) - deletei)
        adj = sp.csr_matrix(([1]*len(row), (row, col)), shape=(N,N))
        adj = adj + adj.T
        adj[adj > 1] = 1
        return adj
    
        

class edge2list():
    def list(adj):
        existedEdges = []
        t = adj.nonzero()
        rows = t[0]
        cols = t[1]
        for i in range(len(rows)):
            existedEdges.append((rows[i], cols[i]))

        return existedEdges
    
    def sett(adj):
        existedEdges = set()
        t = adj.nonzero()
        rows = t[0]
        cols = t[1]
        for i in range(len(rows)):
            existedEdges.add((rows[i], cols[i]))

        return existedEdges


class heap():
    def __init__(self, size=20, element=-10000):
        self.size = size
        self.element = element
        self.h = [element]*self.size

    def push(self, a):
        if a < self.h[0]:
            return

        heapq.heapreplace(self.h, a)

    def sortedlist(self):
        return sorted(self.h, reverse=True)


class Preprocess():
    def loadtxt(filename, type='karate'):
        with open('data/karate/karate_edges_77.txt') as edgesfile, open('data/karate/karate_groups.txt') as labelfile:
            l = []
            for line in labelfile:
                s = line.strip().split('\t')
                l.append(int(s[1]))
            ll = np.zeros((len(l), max(l)))
            for i in range(len(l)):
                ll[i][l[i]-1] = 1

            data = []
            ea = []
            eb = []
            for line in edgesfile:
                s = line.strip().split('\t')
                ea.append(int(s[0]) - 1)
                eb.append(int(s[1]) - 1)
                data.append(1)

            return sp.csr_matrix((data, (ea, eb)), shape=(len(l), len(l))), sp.csr_matrix(np.identity(len(l))), np.array(l)

    def load_data(filename, llc=False):
        ''' load data using nettack api
        args:
            filename: dataset name

        returns:
            _A_obs
            _X_obs
            _z_obs
        '''
        _A_obs, _X_obs, _z_obs = utils.load_npz(filename)
        # print('adj.nnz before: {}'.format(_A_obs.nnz))
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        # print('adj.nnz after: {}'.format(_A_obs.nnz))
        if _X_obs is None:
            row = []
            col = []
            data = []
            for i in range(_A_obs.shape[0]):
                row.append(i)
                col.append(i)
                data.append(1)
            _X_obs = sp.csr_matrix((data, (row, col)), shape=(len(row), len(row)))

        _X_obs = _X_obs.astype('float32')
        if not llc:
            return _A_obs, _X_obs, _z_obs
        lcc = utils.largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:, lcc]
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1

        _X_obs = _X_obs[lcc].astype('float32')
        _z_obs = _z_obs[lcc]
        return _A_obs, _X_obs, _z_obs
    
    def load_split(file):
        with open(file, 'rb') as infile:
            res = pkl.load(infile)
        return res
    def save_split(file, obj):
        with open(file, 'wb') as outfile:
            pkl.dump(obj, outfile)
    
    def split_deep(data, target, dataset='cora'):
        if dataset == 'cora':
            splitfile = 'data/cora-default-split.pkl'
        elif dataset == 'citeseer':
            splitfile = 'data/citeseer-default-split.pkl'
        else:
            print('split dataset name error', dataset)
            exit(0)
        train, val, test = Preprocess.load_split(splitfile)
        # train.extend(val)
        # print(train, type(train), type(test))
        return data[train], data[test], target[train], target[test]
    
    def load_embedding(file, minus=0):
        with open(file) as infile:
            l = next(infile)
            lines = int(l.split(' ')[0])
            res = [0] * lines
            for line in infile:
                try:
                    strs = line.strip('\n').split(' ')
                    if int(strs[0]) == 0:
                        print('index 0', file)
                    res[int(strs[0])-minus] = list(map(float, strs[1:]))
                except IndexError as e:
                    # print('line error', strs[0], lines, len(strs))
                    # raise e
                    res.append(list(map(float, strs[1:])))
            for i in range(len(res)):
                if res[i] == 0:
                    res[i] = [0.0] * len(res[-1])
            ret = np.array(res)
            # print(ret)
            return ret


    def save_data(filename, adj, feature, label):
        ''' save data as npz
        '''
        adj_indices = adj.indices
        adj_indptr = adj.indptr
        adj_data = adj.data
        adj_shape = adj.shape

        attr_data = feature.data
        attr_indices = feature.indices
        attr_indptr = feature.indptr
        attr_shape = feature.shape

        np.savez(filename, adj_indices=adj_indices,
                 adj_indptr=adj_indptr, adj_data=adj_data, adj_shape=adj_shape, attr_data=attr_data, attr_indices=attr_indices, attr_indptr=attr_indptr, attr_shape=attr_shape, labels=label)

    def split_data(_N, _z_obs_1, seed=123, share=(0.1, 0.1)):
        train_share, val_share = share
        unlabeled_share = 1 - train_share - val_share
        np.random.seed(seed)

        split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(_N),
                                                                                     train_size=train_share,
                                                                                     val_size=val_share,
                                                                                     test_size=unlabeled_share,
                                                                                     stratify=_z_obs_1)

        return split_train, split_val, split_unlabeled


# class spa():
    def delete_edges(adj, k=0.5, strat='random'):
        ''' delete edges from a given graph
            args:
                adj: adjacent matrix, sparse
                k: remained size

            returns:
                new_adj: sparse
        '''

        if strat == 'random':
            t = adj.nonzero()
            data = adj.data
            rows = t[0]
            cols = t[1]

            dd = []
            for i in range(len(data)):
                if rows[i] > cols[i]:
                    continue
                dd.append((data[i], rows[i], cols[i]))
            random.shuffle(dd)
            deleted_d = dd[int(len(dd)*k):]
            remained_d = dd[:int(len(dd)*k)]
            rrr = []
            ccc = []
            d = []
            for a, b, c in remained_d:
                ccc.append(b)
                rrr.append(c)
                d.append(a)
        csr = sp.csr_matrix((d, (rrr, ccc)), shape=adj.shape)
        csr = csr + csr.T
        csr[csr > 1] = 1
        return csr, remained_d, deleted_d
    
    def delete_edges_min_tree(adj, k=0.5):
        ree = spa.minconnect(adj)
        csr, _, _2 = spa.delete_edges(adj, k)
        for a, b in ree:
            csr[a, b] = csr[b, a] = 1
        
        return csr, _, _2
    
    def minconnect(adj):
        res = []
        added = set()
        lil = adj.tolil()
        _N = adj.shape[0]
        for i in range(_N):
            if i not in added:
                added.add(i)
            for x in range(_N):
                if lil[i, x] != 0 and x not in added:
                    res.append((i,x))
                    added.add(x)
        
        return res

class test_u():
    def newadj_csr(N):
        d = [1] * N
        row = [random.randint(0, N-1) for i in range(N)]
        col = [random.randint(0, N-1) for i in range(N)]
        return sp.csr_matrix((d, (row, col)), shape = (N,N))


class Eval():
    def acu(predictions, labels):
        preds = np.argmax(predictions, axis=1)
        # precision, recall, fscore, support = score(labels, preds)
        # print('precisions for each class: {}'.format(precision))
        acu = accuracy_score(labels, preds)
        return acu
    
    def f1_score(predictions, labels):
        preds = np.argmax(predictions, axis=1)
        return f1_score(labels, preds, average='micro') + f1_score(labels, preds, average='macro')


# x=np.arange(20,350)

class Matplot():
    def line(data, labels=['type1', 'type2', 'type3']):
        ''' plot line
        args:
            data: [[x1, y1], ...]
        '''
        color = ['r--', 'g--', 'b--']
        for i in range(len(data)):
            plt.plot(data[i][0], data[i][1], color[i], label=labels[i])
        # plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
        # plt.title('The Lasers in Three Conditions')
        plt.xlabel('row')
        plt.ylabel('column')
        plt.legend()
        plt.show()

class Hist():
    def plt(arr, index=0):
        a = []
        for d in arr:
            a.append(d)
        plt.hist(a, bins=100)
        plt.show()

    def subplot(arrs):
        lens = len(arrs)
        # for i in range(len(arrs[0])):
        for j in range(lens):
            # a = []
            # for d in arrs[j]:
            #     a.append(d[i])
            plt.subplot(1,lens,j+1)
            plt.hist(arrs[j], bins=100, density=True)
        plt.xlabel('scores')
        plt.ylabel('count')
        plt.show()

    def subplots(arrs, labels):
        ls = len(arrs)
        f, axs = plt.subplots(2, ls, sharey='row')
        for i in range(ls):
            # ax.plot(x, y)
            # ax1.set_title('Sharing Y axis')
            axs[0, i].hist(arrs[i][0], bins=100)
            axs[1, i].hist(arrs[i][1][1][:-1], bins=arrs[i][1][1], weights=arrs[i][1][0])
            axs[1, i].set_xlabel(labels[i])
        plt.show()
    
class Netplot():
    def plot(labels, edges=None, edges_rm=[]):
        colorm = ['red', 'green', 'blue', 'yellow', 'purple', 'brown']
        shapes = ['s', 'o', '^', 'd', 'v', '<', '>', 'p', 'h', '8']
        G = nx.karate_club_graph()
        pos = []
        with open('data/karate/karate_pos.txt') as file:
            for line in file:
                strs = line.strip().split(',')
                pos.append((float(strs[0].strip()), float(strs[1].strip())))
        colormap = []
        # shapemap = []
        for l in labels:
            colormap.append(colorm[l-1])
            # shapemap.append(shapes[l-1])
        
        for i in range(4):
            colors = [colorm[i] for p, x in enumerate(G.nodes(data = True)) if labels[p] == i+1]
            nx.draw_networkx_nodes(G, pos, node_shape = shapes[i], label='s', node_color=colors, nodelist = [sNode[0] for p, sNode in enumerate(G.nodes(data = True)) if labels[p] == i+1])

        G.remove_edges_from(edges_rm)
        nx.draw_networkx_edges(G, pos, edgelist=edges)
        # nx.draw(G, pos=pos,with_labels=True)
        # nx.draw(G, pos=pos, node_color=colormap, with_labels=True)
        plt.axis('off')
        plt.savefig('/Users/davidhu/Desktop/myfig5.eps', format='eps')
        # plt.show()

        
          

# class Model():
#     def subprocess_GCN(adj, fea, labels, sizes, split_t=None):
#         if split_t == None:
#             print('error params in utils.u.model.subprocess_GCN')
#             exit(-1)
#         sp_train, sp_val, sp_test = split_t

#     def new_gcn(n_An, _X_obs, _Z_obs, sizes, split_train, split_val, gpu_id=None):
#         gcn = GCN.GCN(sizes, n_An, _X_obs, "gcn_orig", gpu_id=gpu_id)
#         return gcn

#     def new_gcn_train(n_An, _X_obs, _Z_obs, sizes, split_train, split_val, gpu_id=None):
#         gcn = GCN.GCN(sizes, n_An, _X_obs, "gcn_orig", gpu_id=gpu_id)
#         gcn.train(split_train, split_val, _Z_obs)

#         return gcn

#     def preds(g_model):
#         logits = g_model.logits.eval(session=g_model.session)
#         predictions = g_model.predictions.eval(session=g_model.session, feed_dict={g_model.node_ids: range(g_model.N)})

#         return predictions

#     def logits(g_model):
#         return g_model.logits.eval(session=g_model.session)

if __name__ == "__main__":
    x1 = [20, 33, 51, 79, 101, 121, 132, 145, 162, 182,
          203, 219, 232, 243, 256, 270, 287, 310, 325]
    y1 = [49, 48, 48, 48, 48, 87, 106, 123, 155, 191,
          233, 261, 278, 284, 297, 307, 341, 319, 341]
    x2 = [31, 52, 73, 92, 101, 112, 126, 140, 153,
          175, 186, 196, 215, 230, 240, 270, 288, 300]
    y2 = [48, 48, 48, 48, 49, 89, 162, 237, 302,
          378, 443, 472, 522, 597, 628, 661, 690, 702]
    x3 = [30, 50, 70, 90, 105, 114, 128, 137, 147, 159, 170,
          180, 190, 200, 210, 230, 243, 259, 284, 297, 311]
    y3 = [48, 48, 48, 48, 66, 173, 351, 472, 586, 712, 804, 899,
          994, 1094, 1198, 1360, 1458, 1578, 1734, 1797, 1892]
    Matplot.line([[x1, y1]])
