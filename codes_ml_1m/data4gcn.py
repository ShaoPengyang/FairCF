#-- coding:UTF-8 --
import os
import numpy as np
import math
import sys
import argparse
import pdb
import pickle
from collections import defaultdict
import time
import pandas as pd

data_dir = './ml-1m'

class MovieLens1M():
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'ratings', 'time'])
        # df = df[df['ratings'] == 5]
        return df


# x:id i:range(len) id->range(len)
# for items 1193 2355 1287-> 0 1 2
def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

# input user0-range item0-range ratings time
def create_user_list(df, user_size):
    # user_size:num of dictionary
    user_list = [dict() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user][row.item] = row.ratings
    return user_list
# user_list []中嵌套 字典。user_list[row.user]索引到对应这个user的字典


# output:user_list:a list whose elements are all dict(save itemid for user u)
# enmurate(list)-> {itemsid1:5,   itemsid2:5......}
def split_train_test(user_list, test_size=0.2, val_size=0.1):
    # 列表存储 对应每个user
    train_user_list = [dict() for u in range(len(user_list))]
    test_user_list = [dict() for u in range(len(user_list))]
    val_user_list = [dict() for u in range(len(user_list))]
    all_user_list = [dict() for u in range(len(user_list))]
    for user, item_dict in enumerate(user_list):
        # Random select | item_dict.keys()=item_ids
        test_item = set([])
        val_item = set([])
        test_item = set(np.random.choice(list(item_dict.keys()),
                                         size=int(len(item_dict) * test_size),
                                         replace=False))
        # pdb.set_trace()
        val_train = set(item_dict.keys()) - test_item
        val_item = set(np.random.choice(list(val_train),
                                        size=int(len(item_dict) * val_size),
                                        replace=False))
        # pdb.set_trace()
        assert len(test_item) > 0, "No test item for user %d" % user
        assert len(val_item) > 0, "No val item for user %d" % user
        for i in test_item:
            test_user_list[user][i] = item_dict[i]
        for i in val_item:
            val_user_list[user][i] = item_dict[i]
        for i in (set(item_dict.keys()) - test_item - val_item):
            train_user_list[user][i] = item_dict[i]
        # pdb.set_trace()
        all_user_list[user].update(item_dict)
    return train_user_list, test_user_list, val_user_list, all_user_list

# train_user_list: list, for each user a set{} stores num of moives
# output[(user1,item1),(user_u,item_i)]
def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item,ratings) for item,ratings in item_set.items()])
    return pair


# --------------------------load feature data-------------------------#
def load_features():
    u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                        encoding='latin-1', parse_dates=True,engine='python')
    # rating 我们是 直接对id=id-1
    users_sex = users['sex']
    users_sex_1 = [0 if i == 'M' else 1 for i in users_sex]
    users_sex_2 = [1 if i == 'M' else 0 for i in users_sex]
    gender_data = np.ascontiguousarray(users_sex_1)
    gender_data_2 = np.ascontiguousarray(users_sex_2)
    pdb.set_trace()
    return gender_data,gender_data_2

def load_item_feature(gender_data,gender_data_2,train_user_list,item_size):
    # 评分1分和评分5分对item的权重影响应该不一样！
    # 男性评分的均值（这个无法体现两个性别） 还是 男女评分的均值（这个好）？
    # 如果全部是女性评分5，label为5，如果全部是男性评分，label为0
    # 5分：为女性所青睐   1分：女性可能不是不喜欢
    # 添加另一个label，为男性评分
    # 出现nan值是因为根据user划分验证集和测试集的时候
    train_item_list = [dict() for i in range(item_size)]
    item_rating_female = [None] * item_size
    item_rating_male = [None] * item_size
    item_female_inds = [None] * item_size
    for user, item_dict in enumerate(train_user_list):
        for item,rating in item_dict.items():
            train_item_list[item][user] = rating

    for item, user_dict in enumerate(train_item_list):
        tmp_list=[]
        tmp_list_2=[]
        tmp_list_3=[]
        for user, rating in user_dict.items():
            # gender_data 010 gender_data_2 101,对应项为0，算总人数已经隐藏在这里了。
            # 会 +1 / +0
            tmp_female = gender_data[user]*rating
            tmp_female_num = gender_data[user]*1
            tmp_male = gender_data_2[user]*rating
            tmp_list.append(tmp_female)
            tmp_list_2.append(tmp_male)
            tmp_list_3.append(tmp_female_num)

        if len(tmp_list) == 0:
            tmp_list.append(0)
        if len(tmp_list_2) == 0:
            tmp_list_2.append(0)
        if len(tmp_list_3) == 0:
            tmp_list_3.append(0)
        # pdb.set_trace()
        item_label_female = np.mean(tmp_list)
        item_label_male= np.mean(tmp_list_2)
        item_inds = np.mean(tmp_list_3)
        item_rating_female[item] = item_label_female
        item_rating_male[item] = item_label_male
        item_female_inds[item] = item_inds
    return item_rating_female,item_rating_male,item_female_inds


def data():
    s = MovieLens1M(data_dir)
    df = s.load()
    '''
    handle data for gcn prepare
    '''
    # xxxid {xxxid:range(len)}
    df['user'] = df['user']-1
    pdb.set_trace()
    df, itemid = convert_unique_idx(df, 'item')
    # pdb.set_trace()
    # userid:{keys:values} values are 0-n int. so we can change dict into list,and user index to represent values
    # index 0 equals to values 0, and so on
    print('Complete assigning unique index to user and item')

    # 6040
    user_size = len(df['user'].unique())
    # 3706
    item_size = len(df['item'].unique())
    print(user_size)
    print(item_size)

    total_user_list = create_user_list(df, user_size)
    train_user_list, test_user_list, val_user_list, all_user_list = split_train_test(total_user_list)

    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    print('Complete spliting items for training and testing')
    pdb.set_trace()

    train_pair = create_pair(train_user_list)
    test_pair = create_pair(test_user_list)
    val_pair = create_pair(val_user_list)

    gender_data,gender_data_2 = load_features()
    item_rating_female,item_rating_male,item_female_inds = load_item_feature(gender_data,gender_data_2,train_user_list,item_size)

    pdb.set_trace()
    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'all_user_list': all_user_list, 'val_user_list': val_user_list,
               'train_pair': train_pair, 'test_pair': test_pair, 'val_pair': val_pair,
               'gender_data':gender_data,'item_rating_female':item_rating_female,
               'item_rating_male':item_rating_male,'item_inds':item_female_inds}
    dirname = './preprocessed/'
    filename = './preprocessed/ml-1m_gcn.pickle'
    '''
    gcn    no r
    gcn1
    gcn2    no r
    gcn3    sqrt  r
    gcn4    r/5
    gcn5  delete rating <=3
    gcn6  4 A/2  5  A 我前一次真是欧皇，这个老难分到每个item都有了（继续降低4得权重）
    7 100 50 25 10
    8 重新提高权重 50 25 10 5
    '''
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def re_save_data():
    u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                        encoding='latin-1', parse_dates=True, engine='python')
    ages_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users_age = users['age']
    users_age_ = [ages_dict[i] for i in users_age]
    users_age_onehot = np.eye(7)[users_age_]
    # pdb.set_trace()
    with open('./preprocessed/ml-1m_gcn_rebuttal.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        all_user_list = dataset['all_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
    print('Load complete')

    train_item_list = [dict() for i in range(item_size)]
    item_age_inds = [None] * item_size
    for user, item_dict in enumerate(train_user_list):
        for item,rating in item_dict.items():
            train_item_list[item][user] = rating
    for item, user_dict in enumerate(train_item_list):
        tmp_list_0, tmp_list_1, tmp_list_2, tmp_list_3, tmp_list_4, tmp_list_5, tmp_list_6 = [], [], [], [], [], [], []
        for user, rating in user_dict.items():
            # gender_data 010 gender_data_2 101,对应项为0，算总人数已经隐藏在这里了。
            # 会 +1 / +0
            tmp_labels = users_age_onehot[user]
            tmp_list_0.append(tmp_labels[0])
            tmp_list_1.append(tmp_labels[1])
            tmp_list_2.append(tmp_labels[2])
            tmp_list_3.append(tmp_labels[3])
            tmp_list_4.append(tmp_labels[4])
            tmp_list_5.append(tmp_labels[5])
            tmp_list_6.append(tmp_labels[6])
            # pdb.set_trace()
        if len(tmp_list_0) == 0:
            tmp_list_0.append(0)
        if len(tmp_list_1) == 0:
            tmp_list_1.append(0)
        if len(tmp_list_2) == 0:
            tmp_list_2.append(0)
        if len(tmp_list_3) == 0:
            tmp_list_3.append(0)
        if len(tmp_list_4) == 0:
            tmp_list_4.append(0)
        if len(tmp_list_5) == 0:
            tmp_list_5.append(0)
        if len(tmp_list_6) == 0:
            tmp_list_6.append(0)
        item_ind_0 = np.mean(tmp_list_0)
        item_ind_1 = np.mean(tmp_list_1)
        item_ind_2 = np.mean(tmp_list_2)
        item_ind_3 = np.mean(tmp_list_3)
        item_ind_4 = np.mean(tmp_list_4)
        item_ind_5 = np.mean(tmp_list_5)
        item_ind_6 = np.mean(tmp_list_6)
        tmp_item_inds = [item_ind_0, item_ind_1, item_ind_2, item_ind_3, item_ind_4, item_ind_5, item_ind_6]
        #
        item_age_inds[item] = tmp_item_inds
    pdb.set_trace()
    gender_data, gender_data_2 = load_features()
    item_rating_female, item_rating_male, item_female_inds = load_item_feature(gender_data, gender_data_2,
                                                                               train_user_list, item_size)

    pdb.set_trace()
    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'all_user_list': all_user_list, 'val_user_list': val_user_list,
               'train_pair': train_pair, 'test_pair': test_pair, 'val_pair': val_pair,
               'gender_data':gender_data,'item_inds':item_female_inds,
               'age_data':users_age_onehot,"item_age_inds":item_age_inds,
               'age_labels':users_age_,
                }
    dirname = './preprocessed/'
    filename = './preprocessed/ml-1m_gcn_re2.pickle'
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    data()
    re_save_data()