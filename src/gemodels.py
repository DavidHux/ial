
import numpy as np
from .utils import utils_nete as utils

class model_i():

    def __init__(self):
        print('basic gemodel, not completed __init__')
        

    def train(self):
        print('basic gemodel, not completed train method')
        

    def getembeddings(self):
        print('basic gemodel, not completed get embedding method')
        return None

    def setAdj(self, newadj):
        print('basic gemodel, not completed set adj method')
        # self.adj = newadj

    def setFeature(self, feature):
        self.features = feature

    def setLabels(self, labels):
        self.labels = labels

    def performance(self):
        print('basic gemodel, not completed performace method')
        return 0


from .models.GCN_n import GCN_n
from .models.GCN_s import GCN_s
from .utils.u import Preprocess, Eval

class gemodel_GCN(model_i):
    def __init__(self, Adj, features, labels, layersize=16, split_t=None, seed=-1, dropout=0.5, sGCN=False):
        # print('GCN model init')
        self.Adj = Adj
        self.features = features
        self.labels = labels

        _N = Adj.shape[0]
        _K = labels.max()+1
        self._Z_obs = np.eye(_K)[labels]
        self.sizes = [layersize, _K]
        self.seed = seed
        self.dropout = dropout

        if sGCN:
            self.GCN = GCN_s
        else:
            self.GCN = GCN_n

        if split_t == None:
            self.split_train, self.split_val, self.split_unlabeled = Preprocess.splitdata(_N, self.labels)
        else:
            assert type(split_t) == tuple and len(split_t) == 3
            self.split_train, self.split_val, self.split_unlabeled = split_t

        adj = utils.preprocess_graph(self.Adj)
        self.model = self.GCN(self.sizes, adj, self.features, "gcn_orig", gpu_id=None, seed=self.seed, params_dict={'dropout': self.dropout})

    def setAdj(self, newadj):
        self.Adj = newadj
        adj_processed = utils.preprocess_graph(self.Adj)
        self.model = self.GCN(self.sizes, adj_processed, self.features, "gcn_orig", gpu_id=None, seed=self.seed, params_dict={'dropout': self.dropout})

    def train(self):
        self.model.train(self.split_train, self.split_val, self._Z_obs, print_info=False)

    def getembeddings(self):
        predictions = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: range(self.model.N)})

        return predictions

    def performance(self, standard='acu', prt=False):
        if standard == 'acu':
            test_pred = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: self.split_unlabeled})
            test_real = self.labels[self.split_unlabeled]
            per = Eval.acu(test_pred, test_real)
            # per = Eval.acu(self.getembeddings(), self.labels)
            if prt:
                print('acu: {}'.format(per))
            return per

        print('err standard for performace')

    def acu(self):
        test_pred = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: self.split_val})
        test_real = self.labels[self.split_val]
        return Eval.acu(test_pred, test_real)
    
    def val_loss(self):
        feed = {self.model.node_ids: self.split_val,
            self.model.node_labels: self._Z_obs[self.split_val]}
        loss_val = self.model.loss.eval(session=self.model.session, feed_dict=feed)
        return loss_val

    def f1_score(self):
        test_pred = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: self.split_val})
        test_real = self.labels[self.split_val]
        return Eval.f1_score(test_pred, test_real)

    # _loss = model.session.run([model.loss], feed)

    def acu_spec_set(self, sp):
        test_pred = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: sp})
        test_real = self.labels[sp]
        return Eval.acu(test_pred, test_real)
    
    def train_acu(self):
        test_pred = self.model.predictions.eval(session=self.model.session, feed_dict={self.model.node_ids: self.split_train})
        test_real = self.labels[self.split_train]
        return Eval.acu(test_pred, test_real)


from multiprocessing import Process, Pool
class Modeltest_GCN():
    '''
    run a GCN model in a sub process, for parallel computing
    '''

    def f(adj, fea, labels, split_t=None, seed=-1, dropout=0.5):
        gcn = gemodel_GCN(adj, fea, labels, split_t=split_t, seed=seed, dropout=dropout)
        gcn.train()
        return (gcn.getembeddings(), gcn.performance())

    def subprocess_GCN(adj, fea, labels, split_t=None, seed=-1, dropout=0.5):
        if split_t == None:
            print('error params in utils.u.model.subprocess_GCN')
            exit(0)
        p = Pool()
        r = p.apply_async(Modeltest_GCN.f, args=(adj, fea, labels, split_t, seed, dropout))
        p.close()
        p.join()
        res = r.get()
        return res

    def f_perf(adj, fea, labels, split_t=None, seed=-1, dropout=0.5, sgc=False):
        gcn = gemodel_GCN(adj, fea, labels, split_t=split_t, seed=seed, dropout=dropout, sGCN=sgc)
        gcn.train()
        return gcn.performance(), gcn.acu(), gcn.f1_score()

    def subprocess_GCN_perf(adj, fea, labels, split_t=None, seed=-1, dropout=0.5, t=1, sgc=False):
        if split_t == None:
            print('error params in utils.u.model.subprocess_GCN')
        p = Pool()
        res_ = []
        for i in range(t):
            r = p.apply_async(Modeltest_GCN.f_perf, args=(adj, fea, labels, split_t, seed, dropout, sgc))
            res_.append(r)
        p.close()
        p.join()
        ret_test = [x.get()[0] for x in res_]
        ret_val = [x.get()[1] for x in res_]
        ret_train = [x.get()[2] for x in res_]
        return (ret_test, ret_val, ret_train)
    
    def single_process_train(adj, feas, labels, split_t, st=None, seed=-1, dropout=0.5, t=5):
        res_val = []
        res_test = []
        for i in range(t):
            gcn = gemodel_GCN(adj, feas, labels, split_t=split_t, seed=seed, dropout=dropout)
            gcn.train()
            if st != None:
                res_val.append(min(gcn.acu_spec_set(st[0]), gcn.acu_spec_set(st[1])))
            else:
                res_val.append(gcn.acu())
            res_test.append(gcn.performance())
        return res_val, res_test    

class submodel_SGC():
    def subprocess_SGC(adj, fea, labels, split_t=None):
        if split_t == None:
            print('error params in utils.u.model.subprocess_GCN')
            exit(0)
            
        p = Pool()
        r = p.apply_async(submodel_SGC.f_SGC, args=(adj, fea, labels, split_t))
        p.close()
        p.join()
        res = r.get()
        return res
    
    def f_SGC(adj, feas, labels, split_t):
        g = gemodel_GCN(adj, feas, labels, split_t=split_t, sGCN=True)
        g.train()
        # print(g.acu())
        W = g.model.W1.eval(session=g.model.session)
        return W

if __name__ == "__main__":
    gemodel_GCN(1,2,3)