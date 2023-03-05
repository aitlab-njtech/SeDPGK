# -*- coding: utf-8 -*-
import torch
import numpy as np

from data.io_dataset import load_dataset
from data.preprocess import to_binary_bag_of_words, remove_underrepresented_classes, \
    eliminate_self_loops, binarize_labels


def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))


def get_dataset(name, data_path, standardize, train_examples_per_class=None, val_examples_per_class=None):
    dataset_graph = load_dataset(data_path)

    # some standardization preprocessing
    if standardize:     # use largest_connected_components
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    graph_adj, node_features, labels = dataset_graph.unpack()
    #labels = binarize_labels(labels)

    # convert to binary bag-of-words feature representation if necessary
    #if not is_binary_bag_of_words(node_features):
        # if _log is not None:
        #     _log.debug(f"Converting features of dataset {name} to binary bag-of-words representation.")
        #node_features = to_binary_bag_of_words(node_features)

    # some assertions that need to hold for all datasets
    # adj matrix needs to be symmetric
    assert (graph_adj != graph_adj.T).nnz == 0
    # features need to be binary bag-of-word vectors
    #assert is_binary_bag_of_words(node_features), f"Non-binary node_features entry!"

    return graph_adj, node_features, labels


def get_train_val_test_split(random_state,labels,train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,train_size=None, val_size=None, test_size=None):

    num_samples = len(labels)                 #初始化样本数量（节点数量），类的数量
    remaining_indices = list(range(num_samples))               #构建与样本数量等长的索引列表
    #划分训练集，验证集，测试集索引长度
    train_examples_per_class=num_samples//10
    val_examples_per_class=num_samples//10
    test_size=num_samples-2*(train_examples_per_class+val_examples_per_class)
    #train_examples_per_class, val_examples_per_class, test_examples_per_class = None, None, None,
    #train_size, val_size = num_samples//5, num_samples//5
    #test_size = num_samples - train_size - val_size
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # 选择训练样本不考虑类的分布
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)#根据size随机不重复选择索引

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    '''if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1'''

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    labels=np.eye(2)[labels.tolist()]
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])





