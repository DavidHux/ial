from u import edge2list
import copy
import random

class distEdge():
    def disturb(self, adj):
        print('unimplemented distEdge disturb')
        exit(0)
    

class distEdge_ran(distEdge):
    def __init__(self, n):
        self.n = n
    
    def disturb(self, adj):
        _N = adj.shape[0]
        existedges = edge2list.sett(adj)

        res = adj.tolil()
        for i in range(self.n):
            a = random.randint(0, _N-1)
            b = random.randint(0, _N-1)
            while (a,b) in existedges:
                a = random.randint(0, _N-1)
                b = random.randint(0, _N-1)
            res[a, b] = res[b, a] = 1
        
        return res.tocsr()
