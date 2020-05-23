#coding=utf-8

import copy
import time
import numpy as np
import random
import scipy.sparse as sp

# from addEdges import *
from modelTrain import *
from candidateGenerate import *
from edgeEval import *
from edgesUpdation import *
from gemodels import *
from u import Preprocess, spa

''' 迭代式边重构方法
'''


class IALGE():
    '''
    '''
    ''' need to be set
    gemodel = None
    edge_strategy = None
    '''

    def __init__(self, adj, features, labels, tao, minedges, randomedgenum=1000, gemodel='GCN', cangen='knn', edgeEval='max', edgeUpdate='easy', early_stop=20, seed=-1, dropout=0.5, deleted_edges=None, completeadj=None, disturbadj_before=None, params=None, dataset=('cora', 1), testindex=1, split_share=(0.1, 0.1), expectEdgeNum=-1, spaceF=10, simtype='node', split_seed=-1, poolnum=2):
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

        split_ss = 123 if split_seed == -1 else split_seed
        _N = self.adj.shape[0]
        self.split_train, self.split_val, self.split_unlabeled = Preprocess.splitdata(_N, self.labels, seed=split_ss, share=self.split_share)
        if edgeEval == 'SGC':
            self.sgc_val = self.split_val[int(len(self.split_val)/2):]
            self.split_val = self.split_val[:int(len(self.split_val)/2)]

        self.split_t = (self.split_train, self.split_val, self.split_unlabeled)

        if gemodel == 'GCN':
            self.gemodel = ialmodel_gcn(self.adj, self.features, self.labels, self.split_t)
        elif isinstance(gemodel, ialmodel):
            self.gemodel = gemodel
        else:
            print('wrong gemodel, expected type ialmodel, actually type {}'.format(type(gemodel)))
            exit(0)

        if completeadj !=  None:
            # testp, valp, trainp = self.final_result(initadj)
            testp, valp, trainp = self.gemodel.multitest(completeadj)
            self.output('complete adj performance test: {}, val: {}, train: {}'.format(testp, valp, trainp), f=True)

        if disturbadj_before !=  None:
            testp, valp, trainp = self.gemodel.multitest(disturbadj_before)
            self.output('disturbed before adj performance test: {}, val: {}, train: {}'.format(testp, valp, trainp), f=True)


        if cangen == 'knn':
            self.cangen = canGen_knn(self.seedEdgeNum, self.poolnum, self.knn, simtype=simtype)
        elif cangen == 'ran':
            self.cangen = canGen_ran(self.randomedgenum, _N)
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
            self.edgeUpdate = edgesUpdate_easy(self.adj, self.features, self.labels, self.split_t, self.edgenumPit2add, self.poolnum, self.subsetnum, self.seed, self.dropout, expectEdgeNum=expectEdgeNum, spaceF=spaceF)
        elif edgeUpdate == 'topK':
            self.edgeUpdate = edgesUpdate_k(self.edgenumPit2add)
        else:
            self.output('edgeUpdation params err')
            exit(0)
        
        self.edgesunadded = set()
        if _N > 5000:
            kf = int((_N*_N) / 10000000)
            for i in range(_N):
                s = random.randint(1, kf)
                for j in range(i+s, _N, kf):
                    self.edgesunadded.add((i,j))
        else:
            for i in range(_N):
                for j in range(i+1, _N):
                    self.edgesunadded.add((i,j))
            
        t = self.adj.nonzero()
        rows = t[0]
        cols = t[1]
        print('prev unadded edges size: {}'.format(len(self.edgesunadded)))
        for i in range(len(rows)):
            self.edgesunadded.discard((rows[i], cols[i]))
        print('after unadded edges size: {}'.format(len(self.edgesunadded)))


    
    def output(self, s, p=True, f=False):
        if p:
            print(s)
        if f:
            self.outfile.write(s+'\n')
            self.outfile.flush()
    
    def deleteedges(self):
        # A_del, _, __= spa.delete_edges(self.adj, k=1)
        print('delete edges begin')
        A_del = sp.csr_matrix(([], ([], [])), shape=self.adj.shape)
        edges2add = set()
        t = self.adj.nonzero()
        rows = t[0]
        cols = t[1]
        for i in range(len(rows)):
            if rows[i] <= cols[i]:
                edges2add.add((rows[i], cols[i]))
            
        init_test_perf, init_val_perf, _ = self.gemodel.multitest(A_del)
        init_performance = best_performance = init_val_perf
        self.output('init performace, test set {}, val set: {}'.format(init_test_perf, init_val_perf), f=True)

        early_stop_best = 0
        early_stop_it = 0
        early_it = 0
        perfs = []
        for i in range(self.tao):
            if self.cg == 'ran':
                embd_ = 0
            else:
                embd_, _ = self.gemodel.singleTrain(A_del)
            cans = list(edges2add)
            edges_p = self.edgeEval.eval(cans, A_del)
            new_adj, new_perf, addededges = self.edgeUpdate.update(edges_p, A_del, p=0.1)
            for e in addededges:
                edges2add.discard(e)
            print('self.edges unadded len: {}'.format(len(edges2add)))

            test_perf, val_perf, train_perf = self.gemodel.multitest(new_adj)
            self.output('time: {}, test res: {}, val res: {}, train res: {}'.format(time.asctime(time.localtime(time.time())), test_perf, val_perf, train_perf), f=True)
            self.output('time: {}, it: {}, performance: {}, init:{}, best:{}, added {} edges'.format(time.asctime(time.localtime(time.time())), i, val_perf, init_performance, best_performance, (new_adj.nnz)/2), f=True)

            if val_perf <= best_performance:
                early_it += 1
                if early_stop_best == 0 and early_it >= self.early_stop:
                    early_stop_it = i
                    early_stop_best = best_performance
                    self.output('\nearly stop at it: {}, performance: {}, init: {}\n'.format(i, best_performance, init_performance), f=True)
                    # break
            else:
                best_performance = val_perf
                best_adj = new_adj
                if early_stop_best == 0:
                    early_it = 0
                
            if early_stop_best != 0 and (new_adj.nnz)/2 >= self.minedges:
                break
            
            A_del = new_adj

        unlabeled_perf, val_perf, train_perf = self.gemodel.multitest(best_adj)
        nnz_final = best_adj.nnz
        self.output('init performace, test set {}, val set: {}, init {} edges'.format(init_test_perf, init_val_perf, 0), f=True)
        self.output('final performace, test set {}, val set: {}, final {} edges, added {} edges'.format(unlabeled_perf, best_performance, nnz_final, (nnz_final)/2), f=True)
        
        Preprocess.savedata('{}_delete_edges_finaladj.npz'.format(self.taskname), best_adj, self.features, self.labels)

        return best_adj, best_performance


    
    def run(self):
        init_test_perf, init_val_perf, _ = self.gemodel.multitest(self.adj)
        init_performance = best_performance = init_val_perf
        self.output('init performace, test set {}, val set: {}'.format(init_test_perf, init_val_perf), f=True)
       
        if self.disturbadj_before != None:
            A_, _ = self.deleteedges()
        else:
            A_ = self.adj
        best_adj = copy.deepcopy(A_)
        Preprocess.savedata('{}_initadj.npz'.format(self.taskname), self.adj, self.features, self.labels)        

         # init_performance = best_performance = 0

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
            cans = self.cangen.cans(A_, embd_, self.edgesunadded)
            t2 = time.time()
            edges_p = self.edgeEval.eval(cans, A_)
            t3 = time.time()
            new_adj, new_perf, addededges = self.edgeUpdate.update(edges_p, A_)
            t4 = time.time()
            for e in addededges:
                self.edgesunadded.discard(e)
            print('self.edges unadded len: {}'.format(len(self.edgesunadded)))

            test_perf, val_perf, train_perf = self.gemodel.multitest(new_adj)
            t5 = time.time()
            print('time consuming: {} {} {} {}'.format(t2-t1, t3-t2, t4-t3, t5-t4))
            self.output('time: {}, test res: {}, val res: {}, train res: {}'.format(time.asctime(time.localtime(time.time())), test_perf, val_perf, train_perf), f=True)
            self.output('time: {}, it: {}, performance: {}, init:{}, best:{}, added {} edges'.format(time.asctime(time.localtime(time.time())), i, val_perf, init_performance, best_performance, (new_adj.nnz-self.adj.nnz)/2), f=True)

            if val_perf <= best_performance:
                early_it += 1
                if early_stop_best == 0 and early_it >= self.early_stop:
                    early_stop_it = i
                    early_stop_best = best_performance
                    self.output('\nearly stop at it: {}, performance: {}, init: {}\n'.format(i, best_performance, init_performance), f=True)
                    # break
            else:
                best_performance = val_perf
                best_adj = new_adj
                if early_stop_best == 0:
                    early_it = 0
                
            if early_stop_best != 0 and (new_adj.nnz-self.adj.nnz)/2 >= self.minedges:
                break
            
            A_ = new_adj

        # unlabeled_perf, val_perf, train_perf = self.final_result(best_adj)
        unlabeled_perf, val_perf, train_perf = self.gemodel.multitest(best_adj)
        nnz_init = self.adj.nnz
        nnz_final = best_adj.nnz
        self.output('init performace, test set {}, val set: {}, init {} edges'.format(init_test_perf, init_val_perf, nnz_init), f=True)
        self.output('final performace, test set {}, val set: {}, final {} edges, added {} edges'.format(unlabeled_perf, best_performance, nnz_final, (nnz_final-nnz_init)/2), f=True)
        
        Preprocess.savedata('{}_finaladj.npz'.format(self.taskname), best_adj, self.features, self.labels)

        return best_adj, best_performance
        
    
    # def test(self, adj):
    #     embds, per = Modeltest_GCN.subprocess_GCN(adj, self.features, self.labels, split_t=self.split_t) #, seed=self.seed, dropout=self.dropout)
    #     return embds, per
    
    # def final_result(self, adj, t=5):

    #     perf_test, perf_val, perf_train = Modeltest_GCN.subprocess_GCN_perf(adj, self.features, self.labels, split_t=self.split_t, t=t)
        
    #     # print('adj performance test res: {}'.format(res))
    #     return np.mean(perf_test), np.mean(perf_val), np.mean(perf_train)
