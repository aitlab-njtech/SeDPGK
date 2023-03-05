import dgl
import copy
import time
import torch

from models.GCN import GCN
from models.GAT import GAT
from models.APPNP import APPNP
from models.MoNet import MoNet
from models.GCNII import GCNII
from utils.metrics import accuracy
from models.GraphSAGE import GraphSAGE
from data.utils import load_tensor_data

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

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
    loss = F.nll_loss(logp[idx_train], labels[idx_train])       #只用有标签的节点进行训练
    acc_train = accuracy(logp[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
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

def clf_predict(clf):
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
    logp = F.log_softmax(logits, dim=1)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    return logp, preds

dataset_list=['ant-1.7', 'camel-1.2', 'camel-1.6', 'jedit-3.2', 'log4j-1.0', 'log4j-1.2', 'lucene-2.0', 'lucene-2.2',
              'lucne-2.4', 'poi-1.5', 'poi-2.0', 'poi-2.5', 'poi-3.0', 'synapse-1.0', 'synapse-1.1', 'synapse-1.2',
              'velocity-1.4', 'velocity-1.6', 'xalan-2.5', 'xalan-2.6', 'xerces-1.2', 'xerces-1.3']
#加载数据并划分训练集、验证集、测试集；其中训练集和验证集为有标签数据，测试集为无标签数据
adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = load_tensor_data(dataset_list[0], 30, 0)
train_data=features[idx_train]
val_data=features[idx_val]
test_data=features[idx_test]
train_labels=labels[idx_train]
val_labels=labels[idx_val]
test_labls=labels[idx_test]
G = dgl.graph((adj_sp.row, adj_sp.col)).to(device=0)
G.ndata['feat'] = features
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())
#初始化教师集成网络中子网络的模型参数
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
            negative_slope=0.2,
            residual=False).to(device=0)
clf3 = GraphSAGE(
            in_feats=features.shape[1],
            n_hidden=128,
            n_classes=labels.max().item() + 1,
            n_layers=2,
            activation=F.relu,
            dropout=0.2,
            aggregator_type='gcn').to(device=0)
clf4 = APPNP(
            g=G,
            in_feats=features.shape[1],
            hiddens=[64],
            n_classes=labels.max().item() + 1,
            activation=F.relu,
            feat_drop=0.5,
            edge_drop=0.5,
            alpha=0.1,
            k=20).to(device=0)
clf5 = MoNet(
            g=G,
            in_feats=features.shape[1],
            n_hidden=64,
            out_feats=labels.max().item() + 1,
            n_layers=3,
            dim=2,
            n_kernels=4,
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
clf_list = [clf1, clf2, clf3, clf4, clf5, clf6]


clf_fit(clf2,[])
clf_fit(clf3,[])
clf_fit(clf4,[])
clf_fit(clf5,[])
clf_fit(clf6,[])

s1,a_pred = clf_predict(clf_list[1])
s2,b_pred = clf_predict(clf_list[2])
s3,c_pred = clf_predict(clf_list[3])
s4,d_pred = clf_predict(clf_list[4])
s5,e_pred = clf_predict(clf_list[5])

wrong_index1 = np.logical_and(a_pred == b_pred, a_pred==c_pred)
wrong_index2 = np.logical_and(a_pred == d_pred, a_pred==e_pred)
