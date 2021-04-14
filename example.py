import os
import shutil
import logging
import argparse
import pandas as pd
import numpy as np
from GBDT.gbdt import GradientBoostingRegressor
from mpyc import *
from mpyc.runtime import mpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.removeHandler(logger.handlers[0])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

train_data = [[1, 5, 20, 1],
            [2, 7, 30, 3],
            [3, 21, 70, 7],
            [4, 30, 60, 8]]
data=[[5, 25, 65]]
# data = [[1, 5, 20]]
train_feather = ['id', 'age', 'weight', 'label']

secnum = mpc.SecInt()   # note SecInt(37);
#secnum = mpc.SecFlt()
def scale_to_int(f):
    if issubclass(secnum, mpc.Integer):
        scale = lambda a: secnum(int(round(a * f)))  # force Python integers
    else:
        scale = lambda a: secnum(float(a))  # force Python floats
    # print("scale = ", np.vectorize(scale))
    return np.vectorize(scale)
    # return scale

f = 2

def plain2secure(plain):
    secure_data = []
    for index in range(len(plain)):
        new_line = []
        for ele in plain[index]:
            ele = scale_to_int(1<<f)(ele)
            new_line.append(ele)
        secure_data.append(new_line)
        # print("secure data:", secure_data)
    return secure_data


def get_data(model):
    dic = {}
    # secure_train_data = plain2secure(train_data)
    secure_train_data = scale_to_int(f)(train_data)
    secure_predict_data = scale_to_int(f)(data)
    dic['regression'] = [pd.DataFrame(secure_train_data, columns=train_feather),
                         pd.DataFrame(secure_predict_data, columns=['id', 'age', 'weight'])]
    # print("dic = ", dic[model])
    return dic[model]


def run(args):
    model = None
    # 获取训练和测试数据
    data = get_data(args.model)[0]
    test_data = get_data(args.model)[1]
    # 创建模型结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    if len(os.listdir('results')) > 0:
        shutil.rmtree('results')
        os.makedirs('results')
    # 初始化模型
    model = GradientBoostingRegressor(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                          min_samples_split=args.count, is_log=args.log, is_plot=args.plot)

    # 训练模型
    mpc.run(mpc.start())
    model.fit(data)
    # 记录日志
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.FileHandler('results/result.log'.format(iter), mode='w', encoding='utf-8'))
    logger.info(data)
    # 模型预测
    model.predict(test_data)
    # 记录日志
    logger.setLevel(logging.INFO)
    logger.info((test_data['predict_value']))
    mpc.run(mpc.shutdown())
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GBDT-Simple-Tutorial')
    parser.add_argument('--model', default='regression', help='the model you want to use',
                        choices=['regression', 'binary_cf', 'multi_cf'])
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--trees', default=10, type=int, help='the number of decision trees')
    parser.add_argument('--depth', default=4, type=int, help='the max depth of decision trees')
    # 非叶节点的最小数据数目，如果一个节点只有一个数据，那么该节点就是一个叶子节点，停止往下划分
    parser.add_argument('--count', default=2, type=int, help='the min data count of a node')
    parser.add_argument('--log', default=False, type=bool, help='whether to print the log on the console')
    parser.add_argument('--plot', default=False, type=bool, help='whether to plot the decision trees')
    args = parser.parse_args()
    run(args)
    #get_data('regression')

    pass
