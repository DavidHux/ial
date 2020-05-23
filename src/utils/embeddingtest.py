from u import loaddataset
from modelTrain import ialmodel_gcn
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LogisticRegression #线性回归模型
from sklearn.metrics import accuracy_score, f1_score
from u import Preprocess
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# def lg(data, target, dataset='cora', default=True):
def lg(dataset='wine', default=True):
    if default:
        # data_train,data_test,target_train,target_test = Preprocess.split_deep(data, target, dataset)
        data_train,data_test,target_train,target_test = loaddataset.load_sk_data_splited(dataset)
    else:
        data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.8,random_state=125)
    x_train,y_train,x_test,y_test = data_train,target_train,data_test,target_test
    print(len(x_train), len(x_test))

    clf = LogisticRegression().fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    acu = accuracy_score(y_test, y_pred)
    return acu


embeddingdir = '/Users/davidhu/Desktop/npz/coradeepwalkemd/'
percents = ['0.25', '0.5', '0.75', '1)']
datasets = ['cora', 'citeseer']
prf = ['init', 'final']

def test():
    print(lg())
    # exit(0)
    adj, feature, label, train, val, test = loaddataset.loadsk('wine', k=0.1)
    split_t = (train, val, test)
    ial = ialmodel_gcn(adj, feature, label, split_t)
    te, va, tr = ial.multitest(adj)
    print(te, va, tr)
    for i in range(1, 20):
        adj = loaddataset.load_k_neibor(feature, i)
        te, va, tr = ial.multitest(adj)
        print(te, va, tr, i)
    # dirs = os.listdir(embeddingdir)
    # for dataset in datasets:
    #     adj, feature, label = Preprocess.loaddata('data/{}-default.npz'.format(dataset), llc=False)
    #     for percent in percents:

    #         for pr in prf:
    #             results = []
    #             sss = '{}-{}-{}'.format(dataset, percent, pr)
    #             print(sss)
    #             for d in dirs:
    #                 if d.find(dataset) == -1 or d.find(percent) == -1 or d.find(pr) == -1:
    #                     continue
    #                 try:
    #                     feas = Preprocess.load_embedding(embeddingdir+d)
    #                     acu = lg(feas, label)
    #                     print(acu)
    #                     results.append(acu)
    #                 except:
    #                     continue
    


if __name__ == "__main__":
    test()