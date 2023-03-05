from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import scipy.sparse as sp
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.MoNet import MoNet
from models.GCNII import GCNII
from dgl.nn.pytorch.conv import SGConv
from models.utils import get_training_config   ##
from data.utils import load_tensor_data,check_writable
from data.get_dataset import get_experiment_config ##
from utils.logger import get_logger
from utils.metrics import accuracy
from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score,f1_score,recall_score,confusion_matrix
#设置数据集，教师模型，运行设备和labelrate，help参数是对此选项作用的简单描述
def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='poi-2.5.csv', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCNII', help='Teacher Model')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=30, help='Label rate')
    return parser.parse_args()

#设置输出结果的路径
def choose_path(conf):
    output_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                     'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    check_writable(output_dir)
    cascade_dir = output_dir.joinpath('cascade')
    check_writable(cascade_dir)
    return output_dir, cascade_dir

#选择模型并进行初始化
def choose_model(conf):
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        if conf['model_name'] == 'GAT':
            num_heads = 8 #8
        else:
            num_heads = 1
        num_layers = 2#1
        num_out_heads = 1 #
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=8,#8
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=2,#2
                          activation=F.relu,
                          dropout=0.2,#0.5
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=features.shape[1],
                      hiddens=[64],#64
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,#0.1
                      k=20).to(conf['device'])  #10
    elif conf['model_name'] == 'MoNet':
        model = MoNet(g=G,
                      in_feats=features.shape[1],
                      n_hidden=64,#64
                      out_feats=labels.max().item() + 1,
                      n_layers=3, #1
                      dim=2,#2
                      n_kernels=4,#3
                      dropout=0.2).to(conf['device'])#0.7
    elif conf['model_name'] == 'SGC':
        model = SGConv(in_feats=features.shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(conf['device'])
    elif conf['model_name'] == 'GCNII':
        model = GCNII(nfeat=features.shape[1],
                      nlayers=conf['layer'],
                      nhidden=conf['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=conf['dropout'],
                      lamda=conf['lamda'],
                      alpha=conf['alpha'],
                      variant=False).to(conf['device'])
    return model


def train(all_logits, dur, epoch):
    t0 = time.time()
    model.train()#将模型设置为训练模式，保证BN层能够用到每一批数据的均值和方差
    optimizer.zero_grad()# 清空过往梯度
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)#Softmax函数dim=0使得每一列所有元素和为1，dim=1使得每一行所有元素和为1；LogSoftmax即Log(Softmax(x))
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[idx_train], labels[idx_train])#计算loss函数:值计算有标签节点的loss值——半监督
    acc_train = accuracy(logp[idx_train], labels[idx_train])
    loss.backward()#反向传播，计算当前梯度
    optimizer.step()#根据梯度更新网络参数
    dur.append(time.time() - t0)
    model.eval()
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
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


def model_train(conf, model, optimizer, all_logits):
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    while epoch < conf['max_epoch']:
        acc_val, loss_val = train(all_logits, dur, epoch)
        epoch += 1
        if acc_val >= best:
            best = acc_val
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt = 0
        else:
            cnt += 1
        if cnt == conf['patience'] or epoch == conf['max_epoch']:
            print("Stop!!!")
            break

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))


def test(conf):
    model.eval()
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, G.edata['a'] = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    acc_test = accuracy(logp[idx_test],labels[idx_test])#[idx_test]
    print("Test set results: loss= {:.4f} acc_test= {:.4f}".format(
        loss_test.item(), acc_test.item()))
    return acc_test, logp


if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())    #创建解析器并传入arg_parse函数
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')#Path.cwd()获取当前路径并进行路径拼接
    conf = get_training_config(config_path, model_name=args.teacher)#将配置文件路径和教师模型作为参数传入，返回训练模型的参数配置
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')#获取数据配置文件路径
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']#获取dataset.conf.yaml文件中的seed参数并存入conf列表
    #选择GPU块数或CPU
    if args.device > 0:
        conf['device'] = torch.device("cuda:" + str(args.device))
    else:
        conf['device'] = torch.device("cpu")
    conf = dict(conf, **args.__dict__)#将配置文件存储为字典格式
    output_dir, cascade_dir = choose_path(conf)#获取保存输出结果路径
    logger = get_logger(output_dir.joinpath('log'))#获取日志信息
    np.random.seed(conf['seed'])#用于生成指定随机数           conf['seed']值为0，每次生成的随机数相同
    torch.manual_seed(conf['seed'])#设置CPU生成随机数的种子，方便下次复现实验结果。
    torch.backends.cudnn.benchmark = False#用于固定网络结构的模型优化，用于提高模型的效率 ，适合于在GPU下使用，且是固定的网络结构
    torch.backends.cudnn.deterministic = True#将这个flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
    # 加载数据集
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['model_name'], conf['dataset'], args.labelrate, conf['device'])
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    torch.set_printoptions(profile="full")      #设置数据显示长度
    np.set_printoptions(threshold=np.inf)       #用于控制Python中小数的显示精度
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    model = choose_model(conf) #选择模型并进行初始化
    #构造优化器对象
    if conf['model_name'] == 'GCNII':
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': conf['wd1']},
            {'params': model.params2, 'weight_decay': conf['wd2']},
        ], lr=conf['learning_rate'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    all_logits = []
    model_train(conf, model, optimizer, all_logits)
    acc_test, logp = test(conf)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()
    output = np.exp(logp.cpu().detach().numpy())
    acc_test = acc_test.cpu().item()
    matrix = confusion_matrix(labels, preds)
    auc=roc_auc_score(labels,preds)
    recall=recall_score(labels,preds)
    acc=accuracy_score(labels,preds)
    TN = matrix[0][0]
    FN = matrix[1][0]
    TP = matrix[1][1]
    FP = matrix[0][1]
    if TN + FP != 0:
        FPR = FP / (TN + FP)
    else:
        FPR = 1
    print(FPR)
    print(recall)
    print(auc)
    np.savetxt(output_dir.joinpath('preds.txt'), preds, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'), labels, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('output.txt'), output, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'), np.array([acc_test]), fmt='%.4f', delimiter='\t')
    if 'a' in G.edata:
        print('Saving Attention...')
        edge = torch.stack((G.edges()[0], G.edges()[1]),0)
        sp_att = sp.coo_matrix((G.edata['a'].cpu().detach().numpy(), edge.cpu()), shape=adj.cpu().size())
        sp.save_npz(output_dir.joinpath('attention_weight.npz'), sp_att, compressed=True)

