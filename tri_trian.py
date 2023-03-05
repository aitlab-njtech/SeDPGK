import numpy as np
import dgl
import copy
import time
import torch
import sklearn
import argparse
from utils.metrics import accuracy
from models.GCN import GCN
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from s_TP import arg_parse
from data.utils import load_tensor_data,check_writable
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.MoNet import MoNet
from models.GCNII import GCNII


class TriTraining:
    def __init__(self, classifier):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]

    def fit(self, L_X, L_y, U_X):
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)
            self.classifiers[i].fit(*sample)  # 分类器分别训练拟合训练集
        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        Li_X, Li_y = [[]] * 3, [[]] * 3  # to save proxy labeled data
        improve = True
        self.iter = 0

        while improve:
            self.iter += 1  # count iterations

            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    Li_X[i] = U_X[U_y_j == U_y_k]  # when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    if l_prime[i] == 0:  # no updated before
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True

            for i in range(3):
                if update[i]:
                    self.classifiers[i].fit(np.append(L_X, Li_X[i], axis=0), np.append(L_y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])

            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement

    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]

    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))

    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)

def train(clf , all_logits, dur, epoch , optimizer):
    t0 = time.time()
    clf.train()#将模型设置为训练模式，保证BN层能够用到每一批数据的均值和方差
    optimizer.zero_grad()# 清空过往梯度
    if clf==clf1:
        logits = clf(G.ndata['feat'])
    elif clf==clf4:
        logits = clf(G.ndata['feat'])
    elif clf==clf2:
        logits, _ = clf(G.ndata['feat'])
    elif clf==clf3:
        logits = clf(G, G.ndata['feat'])
    elif clf==clf5:
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = clf(G.ndata['feat'], pseudo)
    elif clf == clf6:
        logits = clf(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)#Softmax函数dim=0使得每一列所有元素和为1，dim=1使得每一行所有元素和为1；LogSoftmax即Log(Softmax(x))
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[idx_train], labels[idx_train])#计算loss函数:值计算有标签节点的loss值——半监督
    acc_train = accuracy(logp[idx_train], labels[idx_train])
    loss.backward()#反向传播，计算当前梯度
    optimizer.step()#根据梯度更新网络参数
    dur.append(time.time() - t0)
    clf.eval()
    if clf == clf1:
        logits = clf(G.ndata['feat'])
    elif clf == clf4:
        logits = clf(G.ndata['feat'])
    elif clf == clf2:
        logits, _ = clf(G.ndata['feat'])
    elif clf == clf3:
        logits = clf(G, G.ndata['feat'])
    elif clf == clf5:
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = clf(G.ndata['feat'], pseudo)
    elif clf == clf6:
        logits = clf(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    torch.set_printoptions(profile="full")
    #print(logits)
    logp = F.log_softmax(logits, dim=1)

    # we save the logits for visualization later
    all_logits.append(logp.cpu().detach().numpy())#将结果logp存入列表all_logits
    loss_val = F.nll_loss(logp[idx_val], labels[idx_val])
    acc_val = accuracy(logp[idx_val], labels[idx_val])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    print('Epoch %d | Loss: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | Time(s) %.4f' % (
        epoch, loss.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), dur[-1]))
    return acc_val, loss_val

def clf_fit(clf,all_logits):
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    if clf == clf6:
        optimizer = optim.Adam([
            {'params': clf.params1, 'weight_decay': 0.001},
            {'params': clf.params2, 'weight_decay': 0.0005},
        ], lr=0.01)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, clf.parameters()), lr=0.01,
                               weight_decay=0.01)
    while epoch < 500:
        acc_val, loss_val = train(clf , all_logits, dur, epoch , optimizer)
        epoch += 1
        if acc_val >= best:
            best = acc_val
            state = dict([('model', copy.deepcopy(clf.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt = 0
        else:
            cnt += 1
        if cnt == 50 or epoch == 500:
            print("Stop!!!")
            break

    clf.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))

adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data('GCN', 'ant-1.7', 30, 0)

train_data=features[idx_train]
val_data=features[idx_val]
test_data=features[idx_test]

train_labels=labels[idx_train]
val_labels=labels[idx_val]
test_labls=labels[idx_test]             #训练集和验证集为有标签数据，测试集为无标签数据


G = dgl.graph((adj_sp.row, adj_sp.col)).to(device=0)
G.ndata['feat'] = features

clf1 = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=32,
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=0.8).to(device=0)
clf2 = GAT(
            g=G,
            num_layers=2,
            in_dim=features.shape[1],
            num_hidden=8,#8
            num_classes=labels.max().item() + 1,
            heads=([8] * 2) + [1],
            activation=F.relu,
            feat_drop=0.6,
            attn_drop=0.6,
            negative_slope=0.2,     # negative slope of leaky relu
            residual=False).to(device=0)
clf3 = GraphSAGE(
            in_feats=features.shape[1],
            n_hidden=128,
            n_classes=labels.max().item() + 1,
            n_layers=2,#2
            activation=F.relu,
            dropout=0.2,#0.5
            aggregator_type='gcn').to(device=0)
clf4 = APPNP(
            g=G,
            in_feats=features.shape[1],
            hiddens=[64],#64
            n_classes=labels.max().item() + 1,
            activation=F.relu,
            feat_drop=0.5,
            edge_drop=0.5,
            alpha=0.1,#0.1
            k=20).to(device=0)
clf5 = MoNet(
            g=G,
            in_feats=features.shape[1],
            n_hidden=64,#64
            out_feats=labels.max().item() + 1,
            n_layers=3, #1
            dim=2,#2
            n_kernels=4,#3
            dropout=0.2).to(device=0)
clf6 = GCNII(
            nfeat=features.shape[1],
            nlayers=32,
            nhidden=256,
            nclass=labels.max().item() + 1,
            dropout=0.6,
            lamda=0.5,
            alpha=0.1,
            variant=False).to(device=0)


def test(clf):
    if clf == clf1:
        logits = clf(G.ndata['feat'])
    elif clf == clf4:
        logits = clf(G.ndata['feat'])
    elif clf == clf2:
        logits, _ = clf(G.ndata['feat'])
    elif clf == clf3:
        logits = clf(G, G.ndata['feat'])
    elif clf == clf5:
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = clf(G.ndata['feat'], pseudo)
    elif clf == clf6:
        logits = clf(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    return logp


def measure_error(clf_list, X, y, a, b, c, d, e):
    a_pred = test(clf_list[a])
    b_pred = test(clf_list[b])
    c_pred = test(clf_list[c])
    d_pred = test(clf_list[d])
    e_pred = test(clf_list[e])
    wrong_index = np.logical_and(a_pred != y, a_pred == b_pred==c_pred==d_pred==e_pred)
    # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
    return sum(wrong_index) / sum(a_pred == b_pred==c_pred==d_pred==e_pred)



clf_list=[clf1,clf2,clf3,clf4,clf5,clf6]
for clf in clf_list:
    clf_fit(clf,[])

e_prime = [0.5]*6
l_prime = [0]*6
e = [0]*6
update = [False]*6
Li_X, Li_y = [[]]*6, [[]]*6        #to save proxy labeled data
improve = True
iter = 0
while improve:
    iter += 1  # count iterations

    for i in range(6):
        a, b, c, d, e = np.delete(np.array([0, 1, 2, 3, 4, 5]), i)
        update[i] = False
        e[i] = measure_error(G.ndata['feat'][idx_val], labels[idx_val], a, b, c, d, e)
        if e[i] < e_prime[i]:
            U_y_a = test(clf_list[a])[idx_test]
            U_y_b = test(clf_list[b])[idx_test]
            U_y_c = test(clf_list[c])[idx_test]
            U_y_d = test(clf_list[d])[idx_test]
            U_y_e = test(clf_list[e])[idx_test]
            Li_X[i] = test_data[U_y_a == U_y_b == U_y_c == U_y_d == U_y_e]  # when two models agree on the label, save it
            Li_y[i] = test_labls[U_y_a == U_y_b == U_y_c == U_y_d == U_y_e]
            if l_prime[i] == 0:  # no updated before
                l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
            if l_prime[i] < len(Li_y[i]):
                if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                    update[i] = True
                elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                    L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                    Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                    update[i] = True

    for i in range(3):
        if update[i]:
            self.classifiers[i].fit(np.append(L_X, Li_X[i], axis=0), np.append(L_y, Li_y[i], axis=0))
            e_prime[i] = e[i]
            l_prime[i] = len(Li_y[i])
            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement