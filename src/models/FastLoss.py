
import time
import scipy.sparse as sp
import numpy as np
import random


class FastLoss():
    def __init__(self, adj, X, labels, W1, split_train, W2=None, deleted_e=None):
        self.adj_csr = adj.copy()
        self.adj = adj.tolil()
        self.adj_di = (self.adj + sp.eye(adj.shape[0])).tolil()
        self.adj_preprocessed = self.preprocess_graph(self.adj_di).tolil()
        self.adj_temp = self.adj_preprocessed.copy()
        # self.adj_preprocessed_csr = self.adj_preprocessed.tocsr()
        
        self._N = self.adj.shape[0]

        self.X = X
        self.labels = labels

        self.W1 = W1
        self.W2 = W2
        self.W = W1 if W2 == None else W1.dot(W2)
        self.XW = self.X.dot(self.W)


        self.split_train = sorted(split_train)

        self.label_onehot = np.array([np.eye(self.XW.shape[1])[self.labels[i]] for i in self.split_train])
        self.deleted_e = deleted_e
        self.diag = [self.adj_di[i,i] for i in range(self._N)]

        self.rowsum = self.adj_di.sum(1).A1
        self.sqrt_deg = np.power(self.rowsum, -0.5)
        self.degree_mat_inv_sqrt = sp.diags(np.power(self.rowsum, -0.5)).tolil()

        self.time1 = self.time2 = self.time3 = self.time4 = 0
        self.t1 = self.t2 = 0

    def loss4edges(self, edges):
        res = []
        for e in edges:
            res.append(self.compute_new_a_hat(e))
        return res
    
    def loss4edges_del(self, edges):
        res = []
        for e in edges:
            if e[0] == e[1]:
                continue
            if e[1] == -1:
                res.append(self.compute_new_a_hat_del(e[0]))
            else:
                res.append(self.compute_new_a_hat(e[0]))

        return res

    def preprocess_graph(self, adj_):
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5)).tolil()
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized
    
    def loss_dense(self, new_adj):
        A2 = new_adj[self.split_train].dot(new_adj)
        A2XW = A2.dot(self.XW)

        best_wrong_class_logits = (A2XW - 10000 * self.label_onehot).max(1)
        logits_for_correct_class = np.array([A2XW[i][self.labels[x]] for i, x in enumerate(self.split_train)])
        scores = np.sum(logits_for_correct_class - best_wrong_class_logits)
        return scores
    
    def compute_new_a_hat(self, e):
        res = self.adj_temp
        a, b = e
        idxses = []

        for i in e:
            idxs = np.array(self.adj_preprocessed[i].nonzero())[1]
            idxses.append(idxs)

            deg_i_prime = np.power(self.rowsum[i]+1, -0.5)
            for x in idxs:
                if i == x:
                    res[i, x] = self.diag[i] * deg_i_prime * deg_i_prime
                else:
                    res[i, x] = res[x, i] = self.sqrt_deg[x] * deg_i_prime
        res[a, b] = res[b, a] = np.power((self.rowsum[a]+1)*(self.rowsum[b]+1), -0.5)# * (self.adj_di[a,b]+1)

        score = self.loss_dense(res.tocsr())
        # score = self.loss_true(res.tocsr())

        for p, i in enumerate(e):
            idxs = idxses[p]
            for x in idxs:
                self.adj_temp[i, x] = self.adj_temp[x, i] = self.adj_preprocessed[i, x]
        self.adj_temp[a, b] = self.adj_temp[b, a] = self.adj_preprocessed[a, b]
        
        return score
    
    def compute_new_a_hat_del(self, e):
        res = self.adj_temp
        a, b = e
        idxses = []

        for i in e:
            idxs = np.array(self.adj_preprocessed[i].nonzero())[1]
            idxses.append(idxs)

            deg_i_prime = np.power(self.rowsum[i]-1, -0.5)
            for x in idxs:
                if i == x:
                    res[i, x] = self.diag[i] * deg_i_prime * deg_i_prime
                else:
                    res[i, x] = res[x, i] = self.sqrt_deg[x] * deg_i_prime
        res[a, b] = res[b, a] = 0# * (self.adj_di[a,b]+1)

        score = self.loss_dense(res.tocsr())
        # score = self.loss_true(res.tocsr())

        for p, i in enumerate(e):
            idxs = idxses[p]
            for x in idxs:
                self.adj_temp[i, x] = self.adj_temp[x, i] = self.adj_preprocessed[i, x]
        self.adj_temp[a, b] = self.adj_temp[b, a] = self.adj_preprocessed[a, b]
        
        return score

    def loss_new(self, e):
        score = self.compute_new_a_hat(e)
        return score


    def preprocess_graph_opt(self, adj_, degree_mat_inv_sqrt):
        '''deprecated'''
        t1 = time.time()
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        t2 = time.time()
        self.time2 += t2-t1
        return adj_normalized
    
    def compute_new_a2_opt(self, e):
        '''deprecated'''
        a, b = e

        self.adj_di[a,b] += 1 
        self.adj_di[b,a] += 1
        ap = self.degree_mat_inv_sqrt[a,a]
        bp = self.degree_mat_inv_sqrt[b,b]

        self.degree_mat_inv_sqrt[a,a] = np.power(self.rowsum[a]+1, -0.5)
        self.degree_mat_inv_sqrt[b,b] = np.power(self.rowsum[b]+1, -0.5)

        res = self.preprocess_graph_opt(self.adj_di.tocsr(), self.degree_mat_inv_sqrt.tocsr())
        self.degree_mat_inv_sqrt[a,a] = ap
        self.degree_mat_inv_sqrt[b,b] = bp
        self.adj_di[a,b] -= 1  
        self.adj_di[b,a] -= 1

        return res
    
    
    def loss(self, new_adj):
        '''deprecated'''
        A2XW = new_adj.dot(new_adj).dot(self.XW)

        label_onehot = np.array([np.eye(self.XW.shape[1])[self.labels[i]] for i in self.split_train])
        best_wrong_class_logits = (A2XW[self.split_train] - 10000 * label_onehot).max(1)
        logits_for_correct_class = np.array([A2XW[x][self.labels[x]] for x in self.split_train])
        scores = np.sum(logits_for_correct_class - best_wrong_class_logits)
        return scores


    def compute_new_a2(self, e):
        '''deprecated'''
        adj_temp = self.adj_di.copy()
        adj_temp[e[0],e[1]] = adj_temp[e[1],e[0]] = 1
        return self.preprocess_graph(adj_temp)
    
    def compute_new_a2_del(self, e):
        '''deprecated'''
        adj_temp = self.adj_di.copy()
        if adj_temp[e[0], e[1]] == 0:
            print('edge {} not exist'.format(e))
            return 0

        adj_temp[e[0],e[1]] -= 1
        if e[1] != e[0]:
            adj_temp[e[1],e[0]] -= 1
        return self.loss_dense(self.preprocess_graph(adj_temp))
    
    def test_deleted(self):
        res_del = []
        count = 0
        ss = set()
        for _, a, b in self.deleted_e:
            if a >= b:
                continue
            count +=1
            s2 = self.compute_new_a_hat((a, b))
            if count % 100 == 0:
                print('count {}, loss {}'.format(count, s2))
            res_del.append((s2, (a,b)))
            ss.add(s2)
        
        print('final len {}, set len {}'.format(len(res_del), len(ss)))
        sort_res = sorted(res_del, key=lambda x:x[0], reverse=True)
        for i in range(50):
            print(sort_res[i])
    
    def test_remained(self, remained_e):
        res_del = []
        count = 0
        ss = set()
        for _, a, b in remained_e:
            if a >= b:
                continue
            count +=1
            # s2 = self.compute_new_a2_del((a, b))
            s22 = self.compute_new_a_hat_del((a, b))
            if s2 != s22:
                print('error')
            if count % 100 == 0:
                print('count {}, loss {}'.format(count, s2))
            res_del.append((s2, (a,b)))
            ss.add(s2)
        
        print('final len {}, set len {}'.format(len(res_del), len(ss)))
        sort_res = sorted(res_del, key=lambda x:x[0], reverse=True)
        for i in range(50):
            print(sort_res[i])

    def test(self, initset=None, disset=None):
        res_dist = []
        count = 0
        ss = set()
        for e in disset:
            if e[0] >= e[1]:
                continue
            count +=1
            s2 = self.compute_new_a_hat_del(e)
            ss.add(s2)
            # if count % 1000 == 0:
            #     print('count {}, loss {}'.format(count, s2))

            res_dist.append((e, s2))
        
        sorted_res = sorted(res_dist, key=lambda x:x[1], reverse=True)
        print('final len {}, set len {}'.format(len(res_dist), len(ss)))

        # for x, y in sorted_res[:50]:
        #     print('edge {}, score {}, in init set {}'.format(x, y, x in initset))
        ain = 0
        anotin = 0
        a = []
        bres = []
        binx = []
        for i, b in enumerate(sorted_res):
            if i% 100 == 0 and i > 0:
                # print('ain {} not in {}. ratio {}'.format(ain, anotin, ain/100))
                bres.append(anotin/100)
                binx.append(i-100)
                ain = 0
                anotin = 0
            x, y = b
            if x in initset:
                ain += 1
            else:
                a.append(i/len(res_dist))
                # bres.append(i)
                anotin += 1
        # Hist.plt(a)
        binx.append(binx[-1]+100)
        return a, (bres, binx)
        
        

if __name__ == "__main__":
    import sys
    sys.path.insert(0,'/Users/davidhu/Desktop/code/GE-LP/code_new')
    from u import Preprocess, spa, edge2list, Hist
    from gemodels import gemodel_GCN

    import warnings
    import os
    import copy
    from disturbEdges import distEdge_ran


    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    dataset = 'cora'
    # dataset = 'pubmed'
    # dataset = 'polblogs'
    # dataset = 'citeseer'
    percent = 0.85
    stype = 'node'
    share = (0.052, 0.3693) if dataset == 'cora' else (0.0362, 0.3006)
    if dataset == 'pubmed':
        share = (0.003, 0.05)
    _A_obs, feas, labels = Preprocess.loaddata('data/{}.npz'.format(dataset), llc=False)

    _N = _A_obs.shape[0]
    split_train, split_val, split_unlabeled = split_t = Preprocess.splitdata(_N, labels, seed=123, share=share)

    distnum=1000
    dst = distEdge_ran(distnum)
    deletesizes = [1, 0.8, 0.5, 0.2]
    res = []
    labelk = []
    fo = '{}/{}'
    for i in range(len(deletesizes)):
        adj, remained, deleted = spa.delete_edges(_A_obs, k=deletesizes[i])

        adj_d = dst.disturb(adj)
        print('disturb adj, add {} ran edges, prev {}, after {}'.format(len(deleted), adj.nnz, adj_d.nnz))
        initadjset = edge2list.sett(adj)
        disadjset = edge2list.sett(adj_d)     
        g = gemodel_GCN(adj_d, feas, labels, split_t=split_t, sGCN=True)
        g.train()
        print(g.acu())

        W = g.model.W1.eval(session=g.model.session)

        fl = FastLoss(adj_d, feas, labels, W, split_val, deleted_e=deleted)
        bins, dens = fl.test(initset=initadjset, disset=disadjset)
        res.append((bins, dens))
        labelk.append(fo.format(distnum, adj.nnz/2))
    Hist.subplots(res, labelk)
    # fl.test_deleted()
    # fl.test_remained(remained)

