import random
import time
from multiprocessing import Process, Pool
import scipy.sparse as sp
import numpy as np
from .edgeEval import edgeEval_SGC
from .modelTrain import ialmodel_gcn


class recover:
    def __init__(self, adj, features, labels, split_t, poolnum=20):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.split_t = split_t
        self._N = adj.shape[0]
        self.poolnum = poolnum

    def best(self, adj):
        print('unimplemented gene best method')
        exit(0)
    
class recover_sgc(recover):
    def __init__(self, adj, features, labels, split_t, sgc_val, gemodel, poolnum=20, early_s=10, deletenum=10):
        super().__init__(adj, features, labels, split_t, poolnum=poolnum)

        self.sgc_val = sgc_val
        self.early_s = early_s
        self.deletenum = deletenum
        self.gemodel = gemodel
        self.minedges = 1000
        self.adj.eliminate_zeros()

        self.remainedEdges = set()
        self.deletedEdges = set()

        self.edgeEval = edgeEval_SGC(self.adj, self.features, self.labels, self.split_t, self.sgc_val, poolnum=self.poolnum, direction='del')

        t = self.adj.nonzero()
        rows = t[0]
        cols = t[1]
        for i in range(len(rows)):
            if rows[i] >= cols[i]:
                continue
            self.remainedEdges.add((rows[i], cols[i]))



    def run(self, itl):
        init_test_perf, init_performance, train_perf = self.gemodel.multitest(self.adj)
        itl.output('time: {}, init test res: {}, init val res: {}, init train res: {}'.format(time.asctime(time.localtime(time.time())), init_test_perf, init_performance, train_perf), f=True)
        best_performance = init_performance
        best_adj = self.adj
        early_stop_it = (0, 0)
        early_it = 0
        A_= self.adj.copy()
        for i in range(500):
            cans = list(zip(list(self.remainedEdges), [-1] * len(self.remainedEdges)))
            can2 = list(zip(list(self.deletedEdges), [1] * len(self.deletedEdges)))
            cans.extend(can2)
            edges_p = sorted(self.edgeEval.eval(cans, A_), key=lambda x: x[1], reverse=True)
            print(edges_p[0])
            for j in range(self.deletenum):
                if edges_p[j][0][1] == -1:
                    self.deletedEdges.add(edges_p[j][0][0])
                    self.remainedEdges.discard(edges_p[j][0][0])
                    a, b = edges_p[j][0][0]
                    A_[a, b] = A_[b, a] = 0
                else:
                    self.deletedEdges.discard(edges_p[j][0][0])
                    self.remainedEdges.add(edges_p[j][0][0])
                    a, b = edges_p[j][0][0]
                    A_[a, b] = A_[b, a] = 1
            A_.eliminate_zeros()

            test_perf, val_perf, train_perf = self.gemodel.multitest(A_)
            itl.output('time: {}, test res: {}, val res: {}, train res: {}'.format(time.asctime(time.localtime(time.time())), test_perf, val_perf, train_perf), f=True)
            itl.output('time: {}, it: {}, performance: {}, init:{}, best:{}, deleted {} edges'.format(time.asctime(time.localtime(time.time())), i, val_perf, init_performance, best_performance, (self.adj.nnz-A_.nnz)/2), f=True)

            if val_perf <= best_performance:
                early_it += 1
                if early_stop_it[1]  == 0 and early_it >= self.early_s:
                    early_stop_it = (i, best_performance)
                    itl.output('\nearly stop at it: {}, performance: {}, init: {}\n'.format(i, best_performance, init_performance), f=True)
                    break
            else:
                best_performance = val_perf
                best_adj = A_.copy()
                if early_stop_it[1]  == 0:
                    early_it = 0
                
            # if early_stop_it[1]  != 0 and (self.adj.nnz-A_.nnz)/2 >= self.minedges:
            #     break

        unlabeled_perf, val_perf, train_perf = self.gemodel.multitest(best_adj)
        nnz_init = self.adj.nnz
        nnz_final = best_adj.nnz
        itl.output('recover edges, init performace, test set {}, val set: {}, init {} edges'.format(init_test_perf, init_performance, nnz_init), f=True)
        itl.output('recover edges, final performace, test set {}, val set: {}, final {} edges, deleted {} edges'.format(unlabeled_perf, best_performance, nnz_final, (nnz_final-nnz_init)/2), f=True)
        
        # Preprocess.savedata('{}_finaladj.npz'.format(self.taskname), best_adj, self.features, self.labels)

        return best_adj, best_performance


class recover_gene(recover):
    def __init__(self, adj, features, labels, split_t, st=None, initgenenum=500, keepnum=300, discardnum=10, geneK = 0.7, poolnum=20):
        super().__init__(adj, features, labels, split_t, poolnum=poolnum)

        self.initgenenum = initgenenum
        self.keepnum = keepnum
        self.scoredict = {}
        self.historydict = {}
        self.geneK = geneK
        self.st = st
        self.discardnum = discardnum

        self.edges = []
        t = adj.nonzero()
        data = adj.data
        rows = t[0]
        cols = t[1]

        for i in range(len(data)):
            if data[i] != 0:
                self.edges.append((rows[i], cols[i]))
        
        self.lengene = len(self.edges)
        self.initround()

    def initround(self):
        ss = []
        kk = list(range(self.lengene))
        for i in range(self.initgenenum):
            a = (1<<self.lengene) - 1
            random.shuffle(kk)
            for j in range(self.discardnum):
                a ^= 1 << kk[j]
            ss.append(a)
        self.compute_score(ss)

    def f(self, ss_):
        res = {}
        ial = ialmodel_gcn(self.adj, self.features, self.labels, self.split_t)
        for s in ss_:
            if s in self.historydict:
                continue
            adj_ = self.consadj(s)
            score, score_test = ial.singleprocess_train(adj_, self.st)
            res[s] = (score, score_test)
        return res
    
    def consadj(self, s):
        row = []
        col = []
        for i in range(self.lengene):
            if s & (1 << i):
                row.append(self.edges[i][0])
                col.append(self.edges[i][1])
        adj_ = sp.csr_matrix(([1] * len(row), (row, col)), shape=self.adj.shape)
        return adj_

    def compute_score(self, ss):
        size = int(len(ss) / self.poolnum + 1)
        print('len ss {}, size {}'.format(len(ss), size))
        res_ = []
        p = Pool(self.poolnum)
        for i in range(self.poolnum):
            r = p.apply_async(self.f, args=(ss[i*size:i*(size+1)], ))
            res_.append(r)
        p.close()
        p.join()
        for x in res_:
            d = x.get()
            for k, v in d.items():
                self.scoredict[k] = v
                self.historydict[k] = v
    
    def run(self, epoch=200, early=30, itl=None):
        early_stop = 0
        best = (0,(0,0))
        for i in range(epoch):
            ss = self.cross()
            self.compute_score(ss)
            self.eliminate()
            scores0 = [v[0] for k, v in  self.scoredict.items()]
            scores1 = [v[1] for k, v in  self.scoredict.items()]
            meanscore = np.mean(scores0)
            ooo = 'time {}, recover edges round {}, best score {}, test {}, len {}, average {}, test {}'.format(
                time.asctime(time.localtime(time.time())), i, self.bestscore[1][0], self.bestscore[1][1], len(scores0), np.mean(scores0), np.mean(scores1))
            if itl is not None:
                itl.output(ooo, f=True)
            else:
                print(ooo)
            if self.bestscore[1][0] > best[1][0]:
                best= self.bestscore
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= early:
                    break
        return self.consadj(best[0])

    
    def cross(self):
        res = set()
        t = int(self.keepnum * 2 / 4 + 1)
        keys = list(self.scoredict.keys())
        kk = list(range(self.lengene))

        for i in range(t):
            a = keys[random.randint(0, len(keys)-1)]
            b = keys[random.randint(0, len(keys)-1)]
            if b == a:
                b = (b+1) % len(keys)
            res.add(a & b)
            res.add(a | b)
            tempa = a | 0
            tempb = b | 0
            for i in range(self.lengene):
                r = random.random()
                if r < 0.5:
                    xor = (a ^ b) & (1 << i)
                    tempa ^= xor
                    tempb ^= xor
            res.add(tempa)
            res.add(tempb)

            # tempc = a | 0
            # for i in range(self.lengene):
            #     r = random.random()
            #     if r < 0.2:
            #         tempc ^= 1 << i
            # res.add(tempc)

            tt = (1<<self.lengene) - 1
            random.shuffle(kk)
            for j in range(self.discardnum):
                tt ^= 1 << kk[j]
            res.add(a & tt)
            res.add(b & tt)

            # tempc = a | 0
            # for i in range(self.lengene):
            #     if tempc & (1<<i):
            #         r = random.random()
            #         if r < 0.5:
            #             tempc ^= 1 << i
            # res.add(tempc)

            # tempc = a | 0
            # for i in range(self.lengene):
            #     if (~tempc) & (1<<i):
            #         r = random.random()
            #         if r < 0.5:
            #             tempc ^= 1 << i
            # res.add(tempc)
        
        return list(res)
    
    def eliminate(self):
        t = sorted(self.scoredict.items(), key=lambda x: x[1][0], reverse=True)
        self.scoredict = {k:v for k, v in t[:self.keepnum]}
        print('eliminate res {}'.format((t[0][1], t[1][1])))
        self.bestscore = t[0]

