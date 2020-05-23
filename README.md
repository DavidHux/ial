# IAL 

网络结构学习工具

## 使用工具

命令示例如下：

启动一个学习任务
``` bash
python ial.py start --dataset cora

python ial.py start --name test523 --dataset cora --remain 75 --dist true --distsize 10 --disttype random
```

使用库进行实验：

``` python
from itl import IALGE
from src.untils.u import Preprocess, spa


_adj, feas, labels = Preprocess.load_data('data/cora-default.npz', llc=False)
adj_d, deleted = spa.delete_edges(_adj, k=0.1)
adj = Preprocess.adj_add(adj_d, len(deleted))

        
t = IALGE(adj, feas, labels, 100, 1000, 10, cangen='ran', edgeEval='SGC', edgeUpdate='topK',
                seed=123, poolnum=20)

t.run(recover=True)

```

Algorithm:

```
--------
input:
    A:      init adjacent matrix
    F:      init feature matrix
    \tao :  iter times
    n:      edge size to add in one patch
    s:      candidate patch nums in one iter

output:
    A_o:    finaly outputed adj matrix
    embds:  finaly trained embeddings
--------

A_ = A
for i in 1 to \tao:  # iter \tao times(or stop at a fitted time, like early stoping)
    embd_ = gemodel_train(A_, F)
    ress = {}
    for j in 1 to s:
        A_new = addEdges(A_, embd_, n)
        res = gemodel_test(A_new, F)
        ress = ress + <res, A_new>
    A_ = get_next_A(A, A_ ress)

embd_o = gemodel_out(A_, F)
return A_, embd_o
```
