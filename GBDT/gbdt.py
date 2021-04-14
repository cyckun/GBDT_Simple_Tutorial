"""
Created on ：2019/03/28
@author: Freeman, feverfc1994
"""

import abc
import math
import logging
import pandas as pd
from GBDT.decision_tree import Tree
from mpyc.statistics import (mean, variance, stdev, pvariance, pstdev,
                             mode, median, median_low, median_high)
from mpyc.runtime import mpc
import numpy as np

def scale_to_int(f):
    if issubclass(secnum, mpc.Integer):
        scale = lambda a: secnum(int(round(a * f)))  # force Python integers
    else:
        scale = lambda a: secnum(float(a))  # force Python floats
    # print("scale = ", np.vectorize(scale))
    return np.vectorize(scale)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


secnum = mpc.SecFlt()

class SquaresError:

    def __init__(self):
        #self.mpc = mpc.run(mpc.start())
        pass

    def initialize_f_0(self, data):
        #print("type:", type(data['label']))
        # data['f_0'] = data['label'].mean()
        #data['f_0'] = mean((data['label']))
        data['f_0'] = mean(data['label'].values.tolist())
        return mean(data['label'].values.tolist())

    def calculate_residual(self, data, iter):
        print("enter calculate:;")
        res_name = 'res_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        label_list = data['label'].values.tolist()
        f_prev_name_list = data[f_prev_name].values.tolist()
        print("type fo fpreo list", type(f_prev_name_list[0]), f_prev_name)
        new_list = []
        for i in range(len(label_list)):
            ele = mpc.sub(label_list[i], f_prev_name_list[i])
            new_list.append(ele)

        data[res_name] = pd.Series((ele for ele in new_list))


    def update_f_m(self, data, trees, iter, learning_rate, logger):
        f_prev_name = 'f_' + str(iter - 1)
        f_m_name = 'f_' + str(iter)
        data[f_m_name] = data[f_prev_name]
        print(f_m_name)
        print("t;dfd:", type(data[f_m_name].values.tolist()[0]))
        for leaf_node in trees[iter].leaf_nodes:
            # print("xxx:",type(data.loc[leaf_node.data_index, f_m_name]),data.loc[leaf_node.data_index, f_m_name])
            tmp = data.loc[leaf_node.data_index, f_m_name]
            # data.loc[leaf_node.data_index, f_m_name] = mpc.run(mpc.output(tmp.values.tolist()[0])) + learning_rate * leaf_node.predict_value  # cipher and plain
            tmp1 = scale_to_int(2)(learning_rate * leaf_node.predict_value)
            data.loc[leaf_node.data_index, f_m_name] = mpc.add(tmp.values.tolist()[0], tmp1) # cipher and plain
        # 打印每棵树的 train loss
        self.get_train_loss(data['label'], data[f_m_name], iter, logger)
        print("data f_m_nme type:", type(data[f_m_name].values.tolist()[0]))

    def update_leaf_values(self, targets, y):
        tmp = targets.values.tolist()
        return mpc.run(mpc.output(mean(tmp)))
        #return targets.mean()

    def get_train_loss(self, y, f, iter, logger):
        #loss = ((y - f) ** 2).mean()
        loss = mpc.SecInt(1)

        # logger.info(('第%d棵树: mse_loss:%.4f' % (iter, loss)))



class GradientBoostingRegressor:
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        self.loss = SquaresError()
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.trees = {}
        self.f_0 = {}
        self.is_log = is_log
        self.is_plot = is_plot

    def fit(self, data):
        """
        :param data: pandas.DataFrame, the features data of train training
        """
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]  #cao, plain;
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        # print("type pp-- =", type(data))
        self.f_0 = self.loss.initialize_f_0(data)
        # 对 m = 1, 2, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees + 1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/NO.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            # 计算负梯度--对于平方误差来说就是残差
            logger.info(('-----------------------------构建第%d颗树-----------------------------' % iter))
            self.loss.calculate_residual(data, iter)
            target_name = 'res_' + str(iter)
            self.trees[iter] = Tree(data, self.max_depth, self.min_samples_split,
                                    self.features, self.loss, target_name, logger)
            self.loss.update_f_m(data, self.trees, iter, self.learning_rate, logger)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_value'] = data[f_m_name]
