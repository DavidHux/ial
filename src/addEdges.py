

import utils_nete as utils
from u import heap, Sim
from multiprocessing import Pool
from gemodels import gemodel_GCN, Modeltest_GCN
import multiprocessing
import random
import collections
import copy
import numpy as np

class edgeEmbed():
    def unionNode(embeddings, adj):
        pass

class addEdges():

    def edgeReconstruction(self, prevAdj, embeddings, edgenum=20):
        ''' edge reconstruction interface
            args:
                prevAdj: adjacent matrix, N*N 
                embeddings: embedding matrix trained by ge models, N*D size
            
            return:
                newAdj: reconstructed adjacent matrix
        '''
        print('not completed edges reconstruction method')
        pass

class addEdges_random(addEdges):
    def edgeReconstruction(self, prevAdj, embeddings, edgenum=20):
        return [(1, 2), (2, 3)]
    
class addEdges_random_test(addEdges):
    def __init__(self, features, labels, split_train, split_val, split_unlabeled, deleted_edges, seed=-1):
        self.seed = seed
        self.features = features
        self.labels = labels
        self.split_train = split_train
        self.split_val = split_val
        self.split_unlabeled = split_unlabeled
        self.deleted_edges = deleted_edges

    def par(self, es, prevadj):
        tempAdj = copy.deepcopy(prevadj)
        for i, j in es:
            tempAdj[i, j] = 1
            tempAdj[j, i] = 1
        g = gemodel_GCN(tempAdj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=self.seed, dropout=0)
        g.train()
        per = g.performance()
        # print('{}, {} finished'.format(i, j))
        return per

    def edgeReconstruction(self, prevAdj, embds, edgenum=20):

        better = worse = eq = 0
        # multiprocessing.set_start_method('fork')
        p = Pool()
        res = []
        res2 = []
        
        enum = 30

        for e in range(1000):
            es = []
            for f in range(enum):
                a = self.deleted_edges[random.randint(0, len(self.deleted_edges)-1)]
                es.append((a[1], a[2]))
            r = p.apply_async(self.par, args=(es, prevAdj))
            res.append(r)

        for e in range(1000):
            es = []
            for f in range(enum):
                a = random.randint(0, prevAdj.shape[0]-1)
                b = random.randint(0, prevAdj.shape[0]-1)
                es.append((a, b))
            r = p.apply_async(self.par, args=(es, prevAdj))
            res2.append(r)

        p.close()
        p.join()


        g = gemodel_GCN(prevAdj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=self.seed, dropout=0)
        g.train()
        initperformance = g.performance()

        ret = []
        for x in res:
            ret.append(x.get())
        best = 0
        worst = 100
        for a in ret:
            best = max(best, a)
            worst = min(worst, a)
            if a > initperformance:
                better += 1
            elif a < initperformance:
                worse += 1
            else:
                eq += 1
        
        ret2 = []
        for x in res2:
            ret2.append(x.get())
        better2 = worse2 = eq2 = 0
        best2 = 0
        worst2 = 100
        for a in ret2:
            best2 = max(best2, a)
            worst2 = min(worst2, a)
            if a > initperformance:
                better2 += 1
            elif a < initperformance:
                worse2 += 1
            else:
                eq2 += 1
        print('better2: {}, worse2: {}, eq2: {}, best2: {}, worse2: {}, init: {}'.format(better2, worse2, eq2, best2, worst2, initperformance))
        print('better: {}, worse: {}, eq: {}, best: {}, worse: {}, init: {}'.format(better, worse, eq, best, worst, initperformance))
        exit(-1)

        
        

class addEdges_logisticRegression(addEdges):
    def edgeReconstruction(self, prevAdj, embeddings, edgenum=20):
        pass


class addEdges_MLE(addEdges):
    '''Most Likely edges
    return e edges, for each edge return one most like edges

    '''

    def __init__(self, features, labels, split_train, split_val, split_unlabeled, model_num=2, heapsize_per_edge=100, default_model='GCN'):
        self.features = features
        self.labels = labels
        self.split_train = split_train
        self.split_val = split_val
        self.split_unlabeled = split_unlabeled
        self.model_num = model_num
        self.heapsize_per_edge = heapsize_per_edge
        self.default_model = default_model
        self.poolnum = 20

    def getEdges(self, adj):
        existedEdges = []
        t = adj.nonzero()
        rows = t[0]
        cols = t[1]
        for i in range(len(rows)):
            existedEdges.append((rows[i], cols[i]))
        
        return existedEdges
    
    def getembs(self, adj):
        print('generate several embds')
        embeddings = []
        if self.default_model == 'GCN':
            a = utils.preprocess_graph(adj)
            for i in range(self.model_num):
                g = gemodel_GCN(a, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled))
                g.train()
                emb = g.getembeddings()
                embeddings.append(emb)
        else:
            print('ERR: wrong default GE model for add edges')
            exit(-1)

        return embeddings
    
    def embsims(self, edges, embds, l):
        edgeset = set(edges)
        tempedge = embds[edges[l][0]] + embds[edges[l][1]]
        hp = heap(size=self.heapsize_per_edge, element=(-10000, 0, 0))
        mii = 10000
        for ii in range(len(embds)):
            for jj in range(ii, len(embds)):
                if (ii, jj) in edgeset:
                    continue
                edge_i_j = embds[ii] + embds[jj]
                ff = Sim.min_sq(tempedge, edge_i_j)
                hp.push((-ff, ii, jj))
                mii = min(mii, ff)
        
        res = hp.sortedlist()
        # print('min test: mii: {}, res[0]: {}, res[-1]: {}'.format(mii, res[0][0], res[-1][0]))
        print('origin edge: {}, most like: {},{}'.format(edges[l], res[0][1],  res[0][2]))
        return res



    def edgeReconstruction(self, prevAdj, embeddings, edgenum=20):
        return self.generatenewedge_new(embeddings, prevAdj)
        
        print('edge reconstruction start...')
        res = []
        edges_existed = self.getEdges(prevAdj)
        embds = self.getembs(prevAdj)
        for i in range(edgenum):
            ran = random.randint(0, len(edges_existed)-1)
            candidators = collections.defaultdict(int)
            for j in range(len(embds)):
                simedges = self.embsims(edges_existed, embds[j], ran)
                for k in range(len(simedges)):
                    candidators[(simedges[k][1], simedges[k][2])] -= 1
            ll = sorted(candidators.items(), key=lambda x: x[1])
            # print(len(candidators), ll[:5])
            print('en: {}, sorted list: {}, {}, {}, {}'.format(i, ll[0][0], ll[0][1], ll[1][0], ll[1][1]))
            res.append(ll[0][0])

        return res

    def generatenewedge(self, p, ran, embds, edges_existed):
        candidators = collections.defaultdict(int)
        for j in range(len(embds)):
            simedges = self.embsims(edges_existed, embds[j], ran)
            for k in range(len(simedges)):
                candidators[(simedges[k][1], simedges[k][2])] -= 1
        ll = sorted(candidators.items(), key=lambda x: x[1])
        print('edge {} compute finished'.format(p))
        return ll[0][0]

    def getSimEdgesFromNei(self, edgenum, embds, edges_existed, k):
        temp = set()
        e = edges_existed[edgenum]
        # print(e, len(embds))
        simab = [embds[e[1]], embds[e[0]]]
        res = []
        for x in range(2):
            for i in range(len(embds)):
                if e[0] == i or e[1] == i or (x, i) in temp:
                    continue
                res.append((e[x], i, Sim.min_sq(simab[x], embds[i])))
                temp.add((x, i))
                temp.add((i, x))
        
        res = sorted(res, key=lambda x: x[2])
        return res[:k]

    def edgeEva(self, edge, adj, modelnum=20, edgenum=20):
        res = 0
        n = adj.shape[0]
        for i in range(modelnum):
            tempadj = copy.deepcopy(adj)
            for j in range(edgenum):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                tempadj[a, b] = 1
                tempadj[b, a] = 1
            g = gemodel_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled))
            g.train()
            p1 = g.performance()
            tempadj[edge[0], edge[1]] = 1
            tempadj[edge[1], edge[0]] = 1
            g = gemodel_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled))
            g.train()
            p2 = g.performance()
            if p2 >= p1:
                res += 1
        print('edge: {} evaluated {} better score'.format(edge, res))
        return (res, (edge[0], edge[1]))

    def f(self, index, en, adj, embds, edges_existed, knn=20):
        simedges = self.getSimEdgesFromNei(en, embds, edges_existed, knn)
        return simedges
        # res = []
        # for e in simedges:
        #     ev = self.edgeEva(e, adj)
        #     res.append((ev, e))
        # res = sorted(res, key=lambda x: x[0], reverse=True)
        # print('index:{}, edge: {}, eval result:{}'.format(index, edges_existed[en], res[:3]))
        # return (res[0][1][0], res[0][1][1])


    def generatenewedge_new(self, embds, adj, edgenum=20):
        print('edge reconstruction in parallel start..., new')
        edges_existed = self.getEdges(adj)

        ''' get sim edges '''
        # print('''test here''')
        p=Pool(self.poolnum)
        res_simedges = []
        for i in range(edgenum):
            ran = random.randint(0, len(edges_existed)-1)
            r = p.apply_async(self.f, args=(i, ran, adj, embds, edges_existed))
            # r = p.apply_async(self.getSimEdgesFromNei, args=(i, ran, adj, embds, edges_existed))
            res_simedges.append(r)

        p.close()
        p.join()

        ret = []
        for x in res_simedges:
            ret.append(x.get())
        print('sim edges len{}, len 1:{}'.format(len(ret), len(ret[0])))
        
        ''' sim edges eva'''
        p = Pool(self.poolnum)
        res_eval = [[] for i in range(edgenum)]
        for i in range(edgenum):
            for e in ret[i]:
                r = p.apply_async(self.edgeEva, args=(e, adj))
                res_eval[i].append(r)
        
        p.close()
        p.join()

        final_res = []

        ret_2 = [[] for i in range(edgenum)]
        for i in range(edgenum):
            for x in res_eval[i]:
                ret_2[i].append(x.get())
            ret_2[i] = sorted(ret_2[i], key=lambda x: x[0], reverse=True)
            final_res.append(ret_2[i][0][1])

            
        print('edges all finished, generate edges: {}'.format(final_res))

        return final_res
        

    def edgeReconstruction_par(self, prevAdj, embeddings, edgenum=4):
        print('edge reconstruction in parallel start...')
        res = []
        edges_existed = self.getEdges(prevAdj)
        embds = self.getembs(prevAdj)
        p=Pool()
        for i in range(edgenum):
            ran = random.randint(0, len(edges_existed)-1)
            r = p.apply_async(self.generatenewedge, args=(i, ran, embds, edges_existed))
            res.append(r)

        p.close()
        p.join()

        ret = []
        for x in res:
            ret.append(x.get())
        print('edges all finished, generate edges: {}'.format(ret))

        return ret


class addEdges_KNN(addEdges):
    '''get candidators from KNN neighbours
    return e edges, for each edge return one most like edges

    '''

    def __init__(self, features, labels, split_train, split_val, split_unlabeled, model_num=2, heapsize_per_edge=100, default_model='GCN', hyperp = None):
        self.features = features
        self.labels = labels
        self.split_train = split_train
        self.split_val = split_val
        self.split_unlabeled = split_unlabeled
        self.model_num = model_num
        self.heapsize_per_edge = heapsize_per_edge
        self.default_model = default_model

        if hyperp == None:
            e2a, cand, knn, se, edgeevaltimes = (20, 20, 20, 20, 5)
        else:
            e2a, cand, knn, se, edgeevaltimes = hyperp
        self.enum2add = e2a
        self.candinum = cand
        self.knn = knn
        self.subSetEvalNum = se
        self.edgeevaltimes = edgeevaltimes
        self.poolnum = 20

    def getEdges(self, adj):
        existedEdges = []
        t = adj.nonzero()
        rows = t[0]
        cols = t[1]
        for i in range(len(rows)):
            existedEdges.append((rows[i], cols[i]))
        
        return existedEdges
    
    
    def getSimEdgesFromEdgeEmbeds(self, edges, embds, l):
        edgeset = set(edges)
        tempedge = embds[edges[l][0]] + embds[edges[l][1]]
        hp = heap(size=self.heapsize_per_edge, element=(-10000, 0, 0))
        mii = 10000
        for ii in range(len(embds)):
            for jj in range(ii, len(embds)):
                if (ii, jj) in edgeset:
                    continue
                edge_i_j = embds[ii] + embds[jj]
                ff = Sim.min_sq(tempedge, edge_i_j)
                hp.push((-ff, ii, jj))
                mii = min(mii, ff)
        
        res = hp.sortedlist()
        # print('min test: mii: {}, res[0]: {}, res[-1]: {}'.format(mii, res[0][0], res[-1][0]))
        print('origin edge: {}, most like: {},{}'.format(edges[l], res[0][1],  res[0][2]))
        return res


    def getSimEdgesFromNei(self, edgenum, embds, edges_existed, k):
        temp = set()
        e = edges_existed[edgenum]
        # print(e, len(embds))
        simab = [embds[e[1]], embds[e[0]]]
        res = []
        for x in range(2):
            for i in range(len(embds)):
                if e[0] == i or e[1] == i or (x, i) in temp:
                    continue
                res.append((e[x], i, Sim.min_sq(simab[x], embds[i])))
                temp.add((x, i))
                temp.add((i, x))
        
        res = sorted(res, key=lambda x: x[2])
        return res[:k]


    def getSimEdgesLP(self, edgenum, embds, edges_existed):
        # print('embeds shape:{}, {}, type: {}'.format(len(embds), len(embds[0]), type(embds)))
        e = edges_existed[edgenum]

        res1 = embds[e[0]].dot(embds.T)
        res2 = embds[e[1]].dot(embds.T)
        index1 = [(e[0], i) for i in range(len(embds))]
        index2 = [(e[1], i) for i in range(len(embds))]

        t1 = list(zip(res1, index1))
        t2 = list(zip(res2, index2))
        # t1.extend(t2)

        r1 = sorted(t1, key=lambda x:x[0], reverse=True)
        r2 = sorted(t2, key=lambda x:x[0], reverse=True)

        # print('get sim edges lp res:{}'.format(r[:10]))
        r = (r1, r2)
        edgeex = set(edges_existed)
        ret = []
        for rrr in r:
            count = 0
            for x in rrr:
                if x[1][0]==x[1][1] or x[1] in edgeex or (x[1][1], x[1][0]) in edgeex:
                    continue
                ret.append(x[1])
                count += 1
                if count == self.knn:
                    break

        return ret


    def edgeEva(self, edge, adj, modelnum=20, edgenum=20):
        res = 0
        n = adj.shape[0]
        for i in range(modelnum):
            tempadj = copy.deepcopy(adj)
            for j in range(edgenum):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                tempadj[a, b] = 1
                tempadj[b, a] = 1
            g = gemodel_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled))
            g.train()
            p1 = g.performance()
            tempadj[edge[0], edge[1]] = 1
            tempadj[edge[1], edge[0]] = 1
            g = gemodel_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled))
            g.train()
            p2 = g.performance()
            if p2 >= p1:
                res += 1
        print('edge: {} evaluated {} better score'.format(edge, res))
        return (res, (edge[0], edge[1]))


    def edgeEvaNew(self, candidators, adj, edgenum=20):
        res = collections.defaultdict(list)

        try:
            # n = adj.shape[0]
            lencan = len(candidators)
            modelnum = int(2 * self.candinum * self.knn * self.edgeevaltimes / self.poolnum / edgenum)
            # print('modelnum', modelnum)
            for i in range(modelnum):
                addededge = []
                tempadj = copy.deepcopy(adj)
                for j in range(edgenum):
                    a, b = candidators[random.randint(0, lencan-1)]
                    tempadj[a, b] = 1
                    tempadj[b, a] = 1
                    addededge.append((a,b))

                g = gemodel_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled))
                g.train()
                p1 = g.performance()
                
                for e in addededge:
                    res[e].append(p1)
        except BaseException as err:
            print('raised exception edgeEvaNew: {}'.format(err))

        # print('edge: {} evaluated {} better score'.format(edge, res))
        # print('return eval res:{}'.format(res))
        return res


    def f(self, index, en, adj, embds, edges_existed):
        simedges = self.getSimEdgesLP(en, embds, edges_existed)
        return simedges
    
    def subseteval(self, adj, candidators):
        tempadj = copy.deepcopy(adj)
        eset = set()
        for i in range(self.enum2add):
            ran = random.randint(0, len(candidators)-1)
            eset.add(candidators[ran])
            a, b = candidators[ran]
            tempadj[a, b] = 1
            tempadj[b, a] = 1
        # _, p = Modeltest_GCN.subprocess_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=1, dropout=0)
        g = gemodel_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=1, dropout=0)
        g.train()
        p = g.performance()
        return (eset, p)
            
    
    def result(self, pers, adj):
        
        ''' test eval result'''
        listss = []
        evaledgenum = 20
        evalepocnum = 20
        for k,v in pers.items():
            ss = sorted(v, reverse=True)
            listss.append((k, ss[0], np.mean(ss), np.mean(ss[:10])))
        
        rtt = []
        for i in range(3):
            tempadj = copy.deepcopy(adj)
            templistss = sorted(listss, key=lambda x: x[i+1], reverse=True)
            for j in range(evaledgenum):
                a, b = templistss[j][0]
                tempadj[a, b] = 1
                tempadj[b, a] = 1
            _, p = Modeltest_GCN.subprocess_GCN(tempadj, self.features, self.labels, split_t=(self.split_train, self.split_val, self.split_unlabeled), seed=1, dropout=0)
            rtt.append(p)
            print('performance for test {}: {}'.format(i, p))

        maa = 0
        bp = 0
        for i in range(len(rtt)):
            if rtt[i] > maa:
                bp = rtt[i]
                maa = i
        sslist_ = sorted(listss, key=lambda x: x[+1], reverse=True)
        sslist = [x[0] for x in sslist_[:100]]

        # print(sslist)

        p=Pool(self.poolnum)
        res_subset = []
        for i in range(self.subSetEvalNum):
            r = p.apply_async(self.subseteval, args=(adj, sslist))
            res_subset.append(r)

        p.close()
        p.join()

        edgesets_eval = []
        for x in res_subset:
            a = x.get()
            edgesets_eval.append(a)
        eee = sorted(edgesets_eval, key=lambda x: x[1], reverse=True)
        print(eee[0][1], eee[1][1], eee[2][1])

        return eee[0][0]

    def edgeReconstruction(self, adj, embds):
        print('edge reconstruction in parallel start..., new')
        edges_existed = self.getEdges(adj)

        ''' get sim edges '''
        # print('''test here''')
        p=Pool(self.poolnum)
        res_simedges = []
        for i in range(self.candinum):
            ran = random.randint(0, len(edges_existed)-1)
            r = p.apply_async(self.f, args=(i, ran, adj, embds, edges_existed))
            # r = p.apply_async(self.getSimEdgesFromNei, args=(i, ran, adj, embds, edges_existed))
            res_simedges.append(r)

        p.close()
        p.join()

        candiset = set()
        candidators = []
        for x in res_simedges:
            retedges = x.get()
            for e in retedges:
                if e[0] > e[1]:
                    e = (e[1], e[0])
                candiset.add(e)

        candidators = list(candiset)
        print('candidators len{}'.format(len(candiset)))
        
        ''' sim edges eva'''
        p = Pool(self.poolnum)
        res_eval = []
        # psize = int(len(candidators) / poolnum) + 1
        for i in range(self.poolnum):
            r = p.apply_async(self.edgeEvaNew, args=(candidators, adj))
            res_eval.append(r)
        
        p.close()
        p.join()

        final_res = []

        evalres = collections.defaultdict(list)
        s = 0
        try:
            for x in res_eval:
                d = x.get()
                for y, v in d.items():
                    evalres[y].extend(v)

            for k, v in evalres.items():
                s += len(v)
        except BaseException as err:
            print('raised exception evalres: {}'.format(err))

        print('items: {}, totol items: {}'.format(len(evalres), s))

        ressss = self.result(evalres, adj)

            
        # print('edges all finished, generate edges: {}'.format(final_res))

        return ressss
        
