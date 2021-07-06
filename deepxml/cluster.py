#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/24
@author yrh

"""

import os
import numpy as np
import scipy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, hstack
from sklearn.preprocessing import normalize
from logzero import logger
import pandas as pd
from deepxml.data_utils import get_sparse_feature

__all__ = ['build_tree_by_level']

# n2v_emb = pd.read_table("/home/bfs/AttentionXML/data/Amazon-670K/n2vembeddings.emb")

def build_tree_by_level(sparse_data_x, sparse_data_y, mlb, eps: float, max_leaf: int, levels: list, groups_path):
    os.makedirs(os.path.split(groups_path)[0], exist_ok=True)
    logger.info('Clustering')
    sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
    sparse_y = mlb.transform(sparse_labels)
    logger.info('Getting Labels Feature')
    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))

    a = np.zeros(shape=(labels_f.shape[0], 64))
    a = scipy.sparse.csr_matrix(a)
    labels_f_new = hstack([labels_f, a])
    # labels_f_new = np.pad(labels_f, (0, 64), 'constant', constant_values=(0, 0))    # 将label特征矩阵扩充64维
    labels_index = {}  # 存放标签顺序的列表
    index = 0
    for example in sparse_labels:  # 遍历ndarray中的每条样本
        for label in example:  # 遍历每条样本的的标签list
            if label not in labels_index:  # 判断标签是否为第一次出现
                labels_index[label] = index
                index += 1

    # n2v_emb = pd.read_table("../data/Amazon-670K/n2vembeddings.emb")
    emb_list = n2v_emb.values
    emb_list3 = []
    for i in emb_list:
        emb_list2 = i[0].split(" ")
        a_row = []
        for j in emb_list2:
            j = float(j)
            a_row.append(j)
        emb_list3.append(a_row)
    emb = np.array(emb_list3)
    labels_f_new = labels_f_new.tolil()
    for line in emb:
        index = labels_index[str(int(line[0]))]
        labels_f_new[index, -64:] = line[1:]
    labels_f = scipy.sparse.csr_matrix(labels_f_new)  # 将特征矩阵转回csr格式

    logger.info(F'Start Clustering {levels}')
    levels, q = [2 ** x for x in levels], None
    for i in range(len(levels) - 1, -1, -1):
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    while q:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            logger.info(F'Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            logger.info(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
        q = next_q
    logger.info('Finish Clustering')


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n // 2], partition[n // 2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])
