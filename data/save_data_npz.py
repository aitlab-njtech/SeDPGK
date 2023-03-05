import os.path
from pandas import read_csv
import numpy as np
import scipy.sparse as sp
dataset_dir='C:/Users/Ye Yue/Desktop/traditional_feature'
for project in os.listdir(dataset_dir):
    filename = project
    path1 = "C:\\Users\\Ye Yue\\Desktop\\traditional_feature\\"
    path2 = "C:\\Users\\Ye Yue\\Desktop\\traditional+semantic feature\\"
    path3 = "C:\\Users\\Ye Yue\\Desktop\\adj_matrix\\"
    file1 = path1 + filename
    file2 = path2 + filename
    file_g = path3 + filename
    data1 = np.array(read_csv(file1))
    data2 = np.array(read_csv(file2))
    data_g = np.array(read_csv(file_g).fillna(0))
    node1 = data1[:, 2]
    node2 = data2[:, 1]
    node_g = data_g[:, 0]
    data_select1 = data1
    data_select2 = data2
    data_g_select = data_g
    for i in node1:
        if ((i in node_g) & (i in node2)):
            pass
        else:
            index = np.where(node1 == i)[0][0]
            data_select1 = np.delete(data_select1, index, axis=0)
            node1 = data_select1[:, 2]
    for i in node_g:
        if ((i in node1) & (i in node2)):
            pass
        else:
            index = np.where(node_g == i)[0][0]
            data_g_select1 = np.delete(data_g_select, index, axis=0)
            data_g_select = np.delete(data_g_select1, index + 1, axis=1)
            node_g = data_g_select[:, 0]
    for i in node2:
        if ((i in node_g) & (i in node1)):
            pass
        else:
            index = np.where(node2 == i)[0][0]
            data_select2 = np.delete(data_select2, index, axis=0)
            node2 = data_select2[:, 1]
    index1 = 0
    index2 = 0
    for m in node_g:
        index1 = np.where(node_g == m)[0][0]
        if index1 == index2:
            pass
        else:
            data_select1[[index1, index2], :] = data_select1[[index2, index1], :]
            node1 = data_select1[:, 2]
        if index2 < len(node_g) - 1:
            index2 = index2 + 1
        else:
            index2 = index2
    index1 = 0
    index2 = 0
    for m in node_g:
        index1 = np.where(node_g == m)[0][0]
        if index1 == index2:
            pass
        else:
            data_select2[[index1, index2], :] = data_select2[[index2, index1], :]
            node2 = data_select2[:, -1]
        if index2 < len(node_g) - 1:
            index2 = index2 + 1
        else:
            index2 = index2
    adj_matrix = data_g_select[:, 1:len(data_g_select[:, 0]) + 1]
    feature1_matrix = data_select1[:, 3:-1]
    feature2_matrix= data_select2[:,2:221]
    label = data_select1[:, -1]
    for i in range(len(label)):
        if label[i] == 0:
            pass
        else:
            label[i] = 1
    am = sp.coo_matrix(adj_matrix.astype(np.float32)).tocsr()
    fm1 = sp.coo_matrix(feature1_matrix.astype(np.float32)).tocsr()
    fm2 = sp.coo_matrix(feature2_matrix.astype(np.float32)).tocsr()
    data_dict1 = {
        'adj_data': am.data,
        'adj_indices': am.indices,
        'adj_indptr': am.indptr,
        'adj_shape': am.shape,
        'attr_data': fm1.data,
        'attr_indices': fm1.indices,
        'attr_indptr': fm1.indptr,
        'attr_shape': fm1.shape,
        'labels': label
    }
    data_dict2 = {
        'adj_data': am.data,
        'adj_indices': am.indices,
        'adj_indptr': am.indptr,
        'adj_shape': am.shape,
        'attr_data': fm2.data,
        'attr_indices': fm2.indices,
        'attr_indptr': fm2.indptr,
        'attr_shape': fm2.shape,
        'labels': label
    }
    np.savez('C:/Users/Ye Yue/Desktop/dataset_traditonal/' + project + '.npz', **data_dict1)
    np.savez('C:/Users/Ye Yue/Desktop/dataset_semantic/' + project + '.npz', **data_dict2)

