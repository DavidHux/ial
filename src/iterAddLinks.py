#coding=utf-8

import copy
import time

from addEdges import *
from gemodels import *
from u import Preprocess

''' 迭代式边重构方法
'''


class IALGE():
    '''
    '''
    ''' need to be set
    gemodel = None
    edge_strategy = None
    '''

    def __init__(self, adj, features, labels, tao, n, s, gemodel='GCN', edge_Rec='MLE', trainsize=0.5, early_stop=10, seed=-1, dropout=0.5, deleted_edges=None, initadj=None, params=None, dataset=('cora', 1), testindex=1):
        '''
        args:
            adj: init adj matrix, N*N
            feature: N*D
            tao: iter times
            n: candidate patch size
            s: one patch size
            params: (edgenumPit2add, cannumPit, knn, subsetnum) e2a, cand, knn, se
        '''
        self.adj = adj
        self.features = features
        self.tao = tao
        self.n = n
        self.s = s
        self.labels = labels
        self.trainsize = trainsize
        self.early_stop = early_stop
        self.seed = seed
        self.deleted_edges = deleted_edges
        self.dropout = dropout
        if params == None:
            self.params = (20, 20, 20, 20, 5)
        else:
            self.params = params
        
        print('iterAddlinks: params:{} start'.format(self.params))

        self.outfile = open('ial_res_{}_{}_{}_{}.txt'.format(dataset, edge_Rec, self.params, testindex), 'w')

        _N = self.adj.shape[0]
        self.split_train, self.split_val, self.split_unlabeled = Preprocess.splitdata(_N, self.labels)

        if initadj !=  None:
            e, p = self.test(initadj)
            self.output('complete adj performance: {}'.format(p), f=True)

        if gemodel == None:
            self.gemodel = model_i()
        elif gemodel == 'GCN':
            # self.gemodel = gemodel_GCN(self.adj, self.features, self.labels, seed=self.seed, dropout=0)
            self.gemodel = None
        else:
            print('ERR: wrong graph embedding class')
            exit(-1)

        if edge_Rec == 'rand':
            self.edgeRecMethod = addEdges_random()
        elif edge_Rec == 'rand_test':
            self.edgeRecMethod = addEdges_random_test(self.features, self.labels, self.split_train, self.split_val, self.split_unlabeled, self.deleted_edges, self.seed)
        elif edge_Rec == 'MLE':
            self.edgeRecMethod = addEdges_MLE(self.features, self.labels, self.split_train, self.split_val, self.split_unlabeled)
        elif edge_Rec == 'KNN':
            self.edgeRecMethod = addEdges_KNN(self.features, self.labels, self.split_train, self.split_val, self.split_unlabeled, hyperp=self.params)
        else:
            print('ERR: wrong edge reconstruction class')
            exit(-1)

    def setgemodel(self, model):
        if not isinstance(model, gemodel):
            print('model(type: {}) to set is not instance of {}'.format(
                type(model), type(gemodel)))
            exit(0)

        self.gemodel = model
        self.gemodel.setfeature(self.features)
        self.gemodel.setAdj(self.adj)

    def setEdgeRec(self, edgeRec):
        if not isinstance(edgeRec, addEdges):
            print('edge reconstruction method(type: {}) to be set is not instance of {}'.format(
                type(edgeRec), type(addEdges)))

        self.edgeRecMethod = edgeRec
    
    def output(self, s, p=True, f=False):
        if p:
            print(s)
        if f:
            self.outfile.write(s+'\n')
            self.outfile.flush()

    # def
    def run(self):
        A_ = self.adj
        embds1, per1 = self.test(self.adj)
        init_performance = best_performance = per1
        self.output('initial performace: {}, initial edges: {}'.format(init_performance, A_.nnz), f=True)

        early_it = 0
        for i in range(self.tao):
            # self.gemodel.setAdj(A_)
            # embd_ = self.gemodel.getembeddings()
            embd_, _ = self.test(A_)
            new_edges = self.edgeRecMethod.edgeReconstruction(A_, embd_)
            A_temp = copy.deepcopy(A_)
            for a, b in new_edges:
                # print('add edges: {}, {}'.format(a, b))
                A_temp[a, b] = 1
                A_temp[b, a] = 1
            self.output('it: {} add edges, before: {}, after: {}, added: {}'.format(i, A_.nnz, A_temp.nnz, len(new_edges)))
            # self.gemodel.setAdj(A_temp)
            # self.gemodel.train()
            _, per_n = self.test(A_temp)
            self.output('time: {}, it: {}, performance: {}, init:{}, best:{}'.format(time.asctime(time.localtime(time.time())), i, per_n, init_performance, best_performance), f=True)
            if per_n < best_performance:
                early_it += 1
                if early_it >= self.early_stop:
                    self.output('\nearly stop at it: {}, performance: {}, init: {}\n'.format(i, best_performance, init_performance), f=True)
                    break
            else:
                best_performance = per_n
                early_it = 0
                A_ = A_temp


            # ress = []
            # for j in range(n):
            #     A_new = self.edgeRecMethod.edgeReconstruction(A_, embd_)
            #     res = self.gemodel.performance(A_new)
            #     ress.append((res, A_new))
            # A_ = getNextA(self.A, A_ ress)

        self.output('final performace: {}'.format(self.test(A_)), f=True)
        return A_
    
    def test(self, adj):
        embds, per = Modeltest_GCN.subprocess_GCN(adj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=self.seed, dropout=self.dropout)
        return embds, per
        # for i in range(10):
        #     self.gemodel.setAdj(adj)
        #     self.gemodel.train()
        #     print('it: {} performance: {}'.format(i, self.gemodel.performance()))
