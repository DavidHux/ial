import numpy as np


class ialmodel():
    def singleTrain(self, adj):
        print('unimplemented ial model single train')
        exit(0)
    
    def multitest(self, adj, t=5):
        print('unimplemented ial model multi test')
        exit(0)


from .gemodels import Modeltest_GCN

class ialmodel_gcn(ialmodel):
    def __init__(self, adj, features, labels, split_t, sgc=False):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.split_t = split_t
        self.sgc = sgc


    def singleTrain(self, adj):
        embds, per = Modeltest_GCN.subprocess_GCN(adj, self.features, self.labels, split_t=self.split_t) #, seed=self.seed, dropout=self.dropout)
        return embds, per

    
    def multitest(self, adj=None, t=5, seed=-1):
        '''
        returns:
            test, val, train
        '''
        adj_t = adj
        if adj_t == None:
            adj_t = self.adj

        perf_test, perf_val, f1_score = Modeltest_GCN.subprocess_GCN_perf(adj_t, self.features, self.labels, split_t=self.split_t, t=t, seed=seed, sgc=self.sgc)
        return np.mean(perf_test), np.mean(perf_val), np.mean(f1_score)

    def singleprocess_train(self, adj, st=None):
        score_val, score_test = Modeltest_GCN.single_process_train(adj, self.features, self.labels, self.split_t, st=st)
        return np.mean(score_val), np.mean(score_test)
