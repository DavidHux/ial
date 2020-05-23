#coding=utf-8

import copy
import time
import numpy as np
import random

# from addEdges import *
from .modelTrain import *
from .candidateGenerate import *
from .edgeEval import *
from .edgesUpdation import *
from .gemodels import *
from .recoverEdges import *
from .utils.u import Preprocess

''' 迭代式边重构方法
'''


class IALGE():
    '''
    '''
    ''' need to be set
    gemodel = None
    edge_strategy = None
    '''

    def __init__(self, adj, features, labels, tao, minedges, s, randomedgenum=1000, gemodel='GCN', cangen='knn', 
        edgeEval='max', edgeUpdate='topK', early_stop=20, seed=-1, dropout=0.5, deleted_edges=None, 
        disturbadj_before=None, completeadj=None, params=None, dataset=('cora', 1), testindex=1, 
        split_share=(0.1, 0.1), simtype='node', split_seed=-1, poolnum=2):
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
        self.adjlil = self.adj.copy().tolil()
        self.features = features
        self.tao = tao
        self.s = s
        self.labels = labels
        self.early_stop = early_stop
        self.seed = seed
        self.deleted_edges = deleted_edges
        self.dropout = dropout
        self.split_share = split_share
        self.randomedgenum = randomedgenum
        self.cg = cangen
        self.minedges = minedges
        self.disturbadj_before = disturbadj_before


        if params == None:
            self.params = (20, 20, 20, 20, 5)
        else:
            self.params = params
           
        self.edgenumPit2add, self.seedEdgeNum, self.knn, self.subsetnum, self.evalPerEdge = self.params
        self.poolnum = poolnum
  
        print('iterAddlinks: params:{} start'.format(self.params))
        timenow = time.asctime(time.localtime(time.time()))

        self.taskname = 'ial_res_{}_{}_{}_{}_{}'.format(dataset, edgeEval, self.params, testindex, timenow)

        self.outfile = open('{}.txt'.format(self.taskname), 'w')

        self._N = _N = self.adj.shape[0]
        if split_seed == -2:
            self.split_train, self.split_val, self.split_unlabeled = Preprocess.load_split('data/{}-default-split.pkl'.format(dataset[0]))
            print('default split')
        else:
            split_ss = 123 if split_seed == -1 else split_seed
            self.split_train, self.split_val, self.split_unlabeled = sta = Preprocess.split_data(_N, self.labels, seed=split_ss, share=self.split_share)
            Preprocess.save_split('{}.split'.format(self.taskname), sta)

        self.sgc_val = []
        if edgeEval == 'SGC':
            random.shuffle(self.split_val)
            self.sgc_val = self.split_val[int(len(self.split_val)*0.2):]
            # self.split_val = self.split_val[:int(len(self.split_val)*0.2)]

        self.split_t = (self.split_train, self.split_val, self.split_unlabeled)
        print(len(self.split_train), len(self.split_val), len(self.split_unlabeled))

        if gemodel == 'GCN':
            self.gemodel = ialmodel_gcn(self.adj, self.features, self.labels, self.split_t)
        elif isinstance(gemodel, ialmodel):
            self.gemodel = gemodel
        else:
            print('wrong gemodel, expected type ialmodel, actually type {}'.format(type(gemodel)))
            exit(0)

        if completeadj !=  None:
            testp, valp, val_loss = self.gemodel.multitest(completeadj, seed=self.seed)
            self.output('complete adj performance test: {}, val: {}, val_loss: {}'.format(testp, valp, val_loss), f=True)


        if cangen == 'knn':
            self.cangen = canGen_knn(self.seedEdgeNum, self.poolnum, self.knn, simtype=simtype)
        elif cangen == 'ran':
            self.cangen = canGen_ran(self.randomedgenum, self.adj)
        else:
            self.output('cangen params err')
            exit(0)
        
        if edgeEval == 'max':
            self.edgeEval = edgeEval_max(self.adj, self.features, self.labels, self.split_t, self.poolnum, self.knn, self.evalPerEdge, seed=self.seed, dropout=self.dropout)
        elif edgeEval == 'SGC':
            self.edgeEval = edgeEval_SGC(self.adj, self.features, self.labels, self.split_t, self.sgc_val, poolnum=self.poolnum)
        else:
            self.output('edgeeval params err')
            exit(0)

        if edgeUpdate == 'easy':
            self.edgeUpdate = edgesUpdate_easy(self.adj, self.features, self.labels, self.split_t, self.edgenumPit2add, self.poolnum, self.subsetnum, self.seed, self.dropout)
        elif edgeUpdate == 'topK':
            self.edgeUpdate = edgesUpdate_k(self.edgenumPit2add)
        else:
            self.output('edgeUpdation params err')
            exit(0)
        

    
    def output(self, s, p=True, f=False):
        if p:
            print(s)
        if f:
            self.outfile.write(s+'\n')
            self.outfile.flush()

    
    def run(self, recover=False):
        init_test_perf, init_val_perf, val_loss = self.gemodel.multitest(self.adj, seed=self.seed)
        init_performance = best_performance = val_loss
        self.output('init performace, test set {}, val set: {}, loss {}'.format(init_test_perf, init_val_perf, val_loss), f=True)
       
        if recover == True:
            distbefore_test_perf, distbefore_val_perf, _ = self.gemodel.multitest(self.disturbadj_before, seed=self.seed)
            self.output('dist before performace, test set {}, val set: {}'.format(distbefore_test_perf, distbefore_val_perf), f=True)
            # A_ = recover_gene(self.adj, self.features, self.labels, self.split_t, st=(self.sgc_val, self.split_val), poolnum=self.poolnum).run(itl=self)
            sp_val = self.split_val[:int(len(self.split_val)*0.5)]
            sgc_val = self.split_val[int(len(self.split_val)*0.5):]
            split_t_del = (self.split_train, sp_val, self.split_unlabeled)
            print('len split_t_del train {}, val {}, unlabeled {}'.format(len(split_t_del[0]), len(split_t_del[1]), len(split_t_del[2])))
            gemodel_del = ialmodel_gcn(self.adj, self.features, self.labels, split_t_del)
            A_, _ = recover_sgc(self.adj, self.features, self.labels, split_t_del, sgc_val, gemodel_del,
                self.poolnum).run(self)
            Preprocess.savedata('{}_deledadj.npz'.format(self.taskname), A_, self.features, self.labels)        
            test_perf, val_perf, val_loss = self.gemodel.multitest(A_, seed=self.seed)
            self.output('recover outer final performace, test set {}, val set: {}'.format(test_perf, val_perf), f=True)
            # return 0,0
        else:
            A_ = self.adj
        best_adj = copy.deepcopy(A_)
        Preprocess.savedata('{}_initadj.npz'.format(self.taskname), self.adj, self.features, self.labels)        


        early_stop_best = 0
        early_stop_it = 0
        early_it = 0
        perfs = []
        for i in range(self.tao):
            if self.cg == 'ran':
                embd_ = 0
            else:
                embd_, _ = self.gemodel.singleTrain(A_)
            t1 = time.time()
            cans = self.cangen.cans(A_, embd_)
            t2 = time.time()
            edges_p = self.edgeEval.eval(cans, A_)
            t3 = time.time()
            new_adj, new_perf, addededges = self.edgeUpdate.update(edges_p, A_)
            t4 = time.time()

            self.cangen.discard_edges(addededges)

            test_perf, val_perf, val_loss = self.gemodel.multitest(new_adj, seed=self.seed)
            t5 = time.time()
            print('time consuming: {} {} {} {}'.format(t2-t1, t3-t2, t4-t3, t5-t4))
            self.output('time: {}, test res: {}, val res: {}, train res: {}'.format(time.asctime(time.localtime(time.time())), test_perf, val_perf, val_loss), f=True)
            self.output('time: {}, it: {}, performance: {}, init:{}, best:{}, added {} edges'.format(time.asctime(time.localtime(time.time())), i, val_perf, init_performance, best_performance, (new_adj.nnz-self.adj.nnz)/2), f=True)

            if val_loss <= best_performance:
                early_it += 1
                if early_stop_best == 0 and early_it >= self.early_stop:
                    early_stop_it = i
                    early_stop_best = best_performance
                    self.output('\nearly stop at it: {}, performance: {}, init: {}\n'.format(i, best_performance, init_performance), f=True)
                    # break
            else:
                best_performance = val_loss
                best_adj = new_adj
                if early_stop_best == 0:
                    early_it = 0
                
            if early_stop_best != 0 and (new_adj.nnz-self.adj.nnz)/2 >= self.minedges:
                break
            
            A_ = new_adj

        unlabeled_perf, val_perf, val_loss = self.gemodel.multitest(best_adj, seed=self.seed)
        nnz_init = self.adj.nnz
        nnz_final = best_adj.nnz
        self.output('init performace, test set {}, val set: {}, init {} edges'.format(init_test_perf, init_val_perf, nnz_init), f=True)
        self.output('final performace, test set {}, val set: {}, final {} edges, added {} edges'.format(unlabeled_perf, best_performance, nnz_final, (nnz_final-nnz_init)/2), f=True)
        
        Preprocess.savedata('{}_finaladj.npz'.format(self.taskname), best_adj, self.features, self.labels)

        return best_adj, best_performance
        
    
