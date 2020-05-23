
import random
import time
import numpy as np

from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity

from .utils.u import *

class canGen():
    def __init__(self, adj):
        self.adj = adj
        self._N = adj.shape[0]
        self.generate_edges()

    def cans(self, adj, embds, edgesun):
        print('unimplemented cangen cans method')
        exit(0)


    def generate_edges(self):
        t1 = time.time()
        self.edgesunadded = set()
        _N = self._N
        if _N > 5000:
            kf = int(min(_N/2, _N*_N / 2 / self.edgenum))
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
        for i in range(len(rows)):
            self.edgesunadded.discard((rows[i], cols[i]))

        t2 = time.time()
        print('generate unadded set time {}, edges len {}'.format(t2-t1, len(self.edgesunadded)))
    
    def discard_edges(self, edges):
        if self._N <= 5000:
            for e in edges:
                self.edgesunadded.discard(e)
            print('self.edges unadded len: {}'.format(len(self.edgesunadded)))
        else:
            self.generate_edges()


class canGen_ran(canGen):
    def __init__(self, edgenum, adj):
        self.edgenum = edgenum
        super().__init__(adj)
            

    def cans(self, adj, embds):
        eee = list(self.edgesunadded)
        random.shuffle(eee)

        return eee[:self.edgenum]

class canGen_knn(canGen):
    def __init__(self, seedEdgeNum, poolnum, knn, simtype='node'):
        self.seedEdgeNum = seedEdgeNum
        self.poolnum = poolnum
        self.knn = knn
        self.sparsef = 50
        self.simtype = simtype

    def cans(self, adj, embds, edgesun):
        '''generate candidates with KNN method
        params:
            adj: the adj matrix in current iteration
        
        returns:
            edgelist: for example, [(1, 2), (2, 4)]

        '''
        
        edges_existed = edge2list.list(adj)
        p=Pool(self.poolnum)
        res_simedges = []
        for i in range(self.seedEdgeNum):
            ran = random.randint(0, len(edges_existed)-1)
            r = p.apply_async(self.getSimEdgesCOS, args=(ran, embds, edges_existed))
            res_simedges.append(r)

        p.close()
        p.join()

        candiset = set()
        for x in res_simedges:
            retedges = x.get()
            for e in retedges:
                if e[0] > e[1]:
                    e = (e[1], e[0])
                candiset.add(e)

        candidates = list(candiset)
        print('candidates len{}'.format(len(candiset)))
        return candidates
    

    def getSimEdgesCOS(self, edgenum, embds, edges_existed):
        e = edges_existed[edgenum]

        res1 = embds[e[0]].dot(embds.T)
        res2 = embds[e[1]].dot(embds.T)

        norm = np.linalg.norm(embds, axis=1)
        res1 = [res1[i]/(norm[e[0]] * norm[i]) for i in range(len(res1)) ]
        res2 = [res2[i]/(norm[e[1]] * norm[i]) for i in range(len(res2)) ]

        if self.simtype == 'node':
            index1 = [(e[0], i) for i in range(len(embds))]
            index2 = [(e[1], i) for i in range(len(embds))]
        elif self.simtype == 'link':
            # simtype == 'link'
            index1 = [(e[1], i) for i in range(len(embds))]
            index2 = [(e[0], i) for i in range(len(embds))]
        else:
            raise TypeError

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
            for m in range(0, len(rrr), self.sparsef):
                x = rrr[m]
                # for x in rrr:
                if x[1][0]==x[1][1] or x[1] in edgeex or (x[1][1], x[1][0]) in edgeex:
                    continue
                ret.append(x[1])
                count += 1
                if count == self.knn:
                    break

        return ret

if __name__ == "__main__":
    def test():
        ca = canGen_ran(10, 10)
        adj = test_u.newadj_csr(10)
        edges = ca.cans(adj, 0)
        print(edges)
    
    test()
