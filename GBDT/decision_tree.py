"""
Created on ：2019/03/30
@author: Freeman, feverfc1994
"""
import numpy as np
from mpyc.runtime import mpc
from mpyc.statistics import (mean, variance, stdev, pvariance, pstdev,
                             mode, median, median_low, median_high)


class Node:
    def __init__(self, data_index, logger=None, split_feature=None, split_value=None, is_leaf=False, loss=None,
                 deep=None):
        self.loss = loss
        self.split_feature = split_feature
        self.split_value = split_value
        self.data_index = data_index
        self.is_leaf = is_leaf
        self.predict_value = None
        self.left_child = None
        self.right_child = None
        self.logger = logger
        self.deep = deep

    def update_predict_value(self, targets, y):
        self.predict_value = self.loss.update_leaf_values(targets, y)
        self.logger.info(('叶子节点预测值：', self.predict_value))

    def get_predict_value(self, instance):
        if self.is_leaf:
            self.logger.info(('predict:', self.predict_value))
            return self.predict_value
        if instance[self.split_feature] < self.split_value:
            return self.left_child.get_predict_value(instance)
        else:
            return self.right_child.get_predict_value(instance)


class Tree:
    def __init__(self, data, max_depth, min_samples_split, features, loss, target_name, logger):
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = features
        self.logger = logger
        self.target_name = target_name
        self.remain_index = [True] * len(data)
        self.leaf_nodes = []
        self.root_node = self.build_tree(data, self.remain_index, depth=0)

    def build_tree(self, data, remain_index, depth=0):
        """
        此处有三个树继续生长的条件：
            1: 深度没有到达最大, 树的深度假如是3， 意思是需要生长成3层, 那么这里的depth只能是0, 1 
                所以判断条件是 depth < self.max_depth - 1
            2: 点样本数 >= min_samples_split
            3: 此节点上的样本的 target_name 值不一样（如果值 一样说明已经划分得很好了，不需要再分）
        """
        now_data = data[remain_index]
        # print("now_data:", now_data, remain_index)
        now_data_list = now_data.values.tolist()
        # print("now lsit = ", now_data_list)
        sort_list = mpc.sorted(now_data_list[0])
        # await mpc.barrier()
        unique_flag = mpc.eq(sort_list[0], sort_list[len(sort_list) - 1])
        result = mpc.run(mpc.output(unique_flag))
        #await mpc.barrier()
        #print("11111:", result, type(result))
        flag1 = (depth < self.max_depth - 1)
        flag2 = (len(now_data) >= self.min_samples_split)
        flag3 = (result == 0)
        # print('flg:', flag1, flag2, flag3)
        if  flag1 and flag2 and flag3:
            #print("enter new branch")

            se = None
            split_feature = None
            split_value = None
            left_index_of_now_data = None
            right_index_of_now_data = None
            self.logger.info(('--树的深度：%d' % depth))
            for feature in self.features:
                self.logger.info(('----划分特征：', feature))
                # print("feather:", self.features)
                feature_list = now_data[feature].values.tolist()
                tmp_list = mpc.sorted(feature_list)   #
                feature_new_list = []
                for i in range(len(tmp_list)-1):
                    if (mpc.run(mpc.output(mpc.eq(tmp_list[i], tmp_list[i+1]))) == 0):
                        feature_new_list.append(tmp_list[i])
                feature_new_list.append(tmp_list[len(tmp_list)-1])  # add last element;
                #feature_values = now_data[feature].unique()
                #print(",11", feature_values)
                #print('len of f newlist;', len(feature_new_list))
                for fea_val in feature_new_list:
                    #print("new frahte varleu:")
                    # 尝试划分
                    #left_index = list(now_data[feature] < fea_val)
                    #right_index = list(now_data[feature] >= fea_val)
                    left_index = []
                    right_index = []
                    for ele in feature_list:
                        if (mpc.run(mpc.output(mpc.lt(ele, fea_val))) == 1):
                            left_index.append(True)
                            right_index.append(False)
                        else:
                            left_index.append(False)
                            right_index.append(True)
                    #print("\n\n*****************left index", left_index)
                    #print("*****************rith index", right_index)
                    tmp = now_data[left_index][self.target_name]
                    #print("ttttmp: ", tmp)
                    left_se = calculate_se(tmp)
                    right_se = calculate_se(now_data[right_index][self.target_name])
                    sum_se = left_se + right_se
                    #self.logger.info(('------划分值:%.3f,左节点损失:%.3f,右节点损失:%.3f,总损失:%.3f' %
                                      # (fea_val, left_se, right_se, sum_se)))
                    if se is None or sum_se < se:
                        split_feature = feature
                        split_value = fea_val
                        se = sum_se
                        left_index_of_now_data = left_index
                        right_index_of_now_data = right_index
            self.logger.info(('--最佳划分特征：', split_feature))
            self.logger.info(('--最佳划分值：', split_value))
            node = Node(remain_index, self.logger, split_feature, split_value, deep=depth)
            """ 
            trick for DataFrame, index revert
            下面这部分代码是为了记录划分后样本在原始数据中的的索引
            DataFrame的数据索引可以使用True和False
            所以下面得到的是一个bool类型元素组成的数组
            利用这个数组进行索引获得划分后的数据
            """
            left_index_of_all_data = []
            #print("remian_dindex:", remain_index)
            for i in remain_index:
                if i:
                    if left_index_of_now_data[0]:
                        left_index_of_all_data.append(True)
                        del left_index_of_now_data[0]
                    else:
                        left_index_of_all_data.append(False)
                        del left_index_of_now_data[0]
                else:
                    left_index_of_all_data.append(False)

            right_index_of_all_data = []
            for i in remain_index:
                if i:
                    if right_index_of_now_data[0]:
                        right_index_of_all_data.append(True)
                        del right_index_of_now_data[0]
                    else:
                        right_index_of_all_data.append(False)
                        del right_index_of_now_data[0]
                else:
                    right_index_of_all_data.append(False)
            #print("left_idnex_fllda:", left_index_of_all_data)
            #print("left_idnex_fllda:", right_index_of_all_data)
            print("\n\n")
            node.left_child = self.build_tree(data, left_index_of_all_data, depth + 1)
            node.right_child = self.build_tree(data, right_index_of_all_data, depth + 1)
            return node
        else:
            node = Node(remain_index, self.logger, is_leaf=True, loss=self.loss, deep=depth)
            if len(self.target_name.split('_')) == 3:
                label_name = 'label_' + self.target_name.split('_')[1]
            else:
                label_name = 'label'
            node.update_predict_value(now_data[self.target_name], now_data[label_name])
            self.leaf_nodes.append(node)
            return node

secnum = mpc.SecInt()

def calculate_se(label): # square error
    #print("label:", label)
    new_list = label.values.tolist()
    #print("new list:", new_list)
    if len(new_list) > 0:
        mean_value = mean(new_list)
        #mean = label.mean()
        se = secnum(0)
        #for y in label:
            #se += (y - mean) * (y - mean)
        for ele in new_list:
            tmp = mpc.sub(ele, mean_value)
            tmp = mpc.mul(tmp, tmp)
            se = mpc.add(se, tmp)
        return mpc.run(mpc.output(se))   # should return cipher, deal it lator;
    else:
        return 0
