import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from shutil import copyfile
import torch.optim as optim
from torch.nn.init import xavier_normal, xavier_uniform
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score,roc_curve,auc
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import torch.autograd as autograd
import argparse
import pdb
import pickle
from collections import defaultdict
import time
import copy
from models import *
user_num = 6040
item_num = 3706
factor_num = 32
batch_size = 4096*8
lamada = 0.01

def hr_ndcg(indices_sort_top,index_end_i,top_k):
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log2(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1]
    count=0
    for item_id in indices_sort_top:
        if item_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log2(count+2)
        count+=1
        if count==top_k:
            break
    try:
        hr_t=hr_topK/max_hr
    except:
        pdb.set_trace()

    ndcg_t=ndcg_topK/max_ndcg
    return hr_t,ndcg_t

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    # 返回ary中按下标的升序。然后取最后50的在ary中的下标
    indices = np.argpartition(flat, -n)[-n:]
    # pdb.set_trace()
    # 1 从flat中取出对应的评分，然后取负，取负以后排序，得到top50的对于indices的排序的下标
    # indices重新选取，得到排过顺序的top50的id
    indices = indices[np.argsort(-flat[indices])]
    # pdb.set_trace()
    # 输出 top50的数组下item_i_list中的排序
    return np.unravel_index(indices, ary.shape)

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
    y_test = np.asarray(y_test).squeeze()
    y_pred= np.asarray(y_pred).squeeze()
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_pred, average=average)

# find a normal one from other dataset, this have changed for 20210718
def metrics(preds_list,labels_list,probs_list,dataset_len,correct):
    cat_preds_list = preds_list
    cat_labels_list = labels_list
    # cat_probs_list = probs_list
    # cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    # cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = 0
    # AUC = roc_auc_score(cat_labels_list,cat_probs_list,average="micro")
    # AUC = roc_auc_score(cat_labels_list, cat_probs_list)
    # fpr, tpr, thresholds = roc_curve(cat_labels_list, cat_probs_list, pos_label=1)
    # AUC = auc(fpr, tpr)
    acc = 100. * correct / dataset_len
    # pdb.set_trace()
    f1_micro = f1_score(cat_labels_list, cat_preds_list, average='micro')
    f1_macro = f1_score(cat_labels_list, cat_preds_list, average='macro')
    return AUC, acc, f1_micro, f1_macro

def metrics_gender(preds_list,labels_list,probs_list,dataset_len,correct):
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    # pdb.set_trace()
    # AUC = roc_auc_score(cat_labels_list,cat_probs_list,average="micro")
    AUC = roc_auc_score(cat_labels_list, cat_probs_list)
    # fpr, tpr, thresholds = roc_curve(cat_labels_list, cat_probs_list, pos_label=1)
    # AUC = auc(fpr, tpr)
    acc = 100. * correct / dataset_len
    f1_micro = f1_score(cat_labels_list, cat_preds_list, average='binary')
    f1_macro = f1_score(cat_labels_list, cat_preds_list, average='binary')
    return AUC, acc, f1_micro, f1_macro

def fairness_K_re2(user_embedding,item_embedding,topk,test_user_list,all_user_list,gender_data):
    '''
    compute rethinking_fairness@K
    '''
    user_e = user_embedding.cpu().detach().numpy()
    item_e = item_embedding.cpu().detach().numpy()

    all_prediction = np.matmul(user_e, item_e.T)
    set_list_all = set(range(item_num))
    item_male = {}
    item_female = {}
    for i in range(item_num):
        item_female[i] = []
        item_male[i] = []

    for user in range(user_num):
        item_i_list = copy.deepcopy(list(test_user_list[user].keys()))
        index_end_i = len(item_i_list)
        item_j_list = list(set_list_all - set(all_user_list[user].keys()))
        item_i_list.extend(item_j_list)
        prediction = all_prediction[user][item_i_list]
        # pdb.set_trace()
        indices1 = largest_indices(prediction, topk)
        # 返回的是 prediction中的排序前50的index
        # 然后正常计算 hr ndcg的时候，选取的是 indices(在item_i_list中的下标)和 index_end_i
        # index_end_i是 item_i_list 前多少个是属于 测试集。
        indices1 = list(indices1[0])
        item_i_list = np.array(item_i_list)
        top_k_index = item_i_list[indices1]
        for i in top_k_index:
            if gender_data[user] == 0:
                item_male[i].append(all_prediction[user][i])
            else:
                item_female[i].append(all_prediction[user][i])
    fairness_rethink_K = 0
    for i in range(item_num):
        '''
        situation: item recommended to male and not to females.
        '''
        # if len(item_male[i]) == 0 or len(item_female[i]) == 0:
        #     extra = 0
        if len(item_male[i]) == 0 and len(item_female[i]) == 0:
            extra = 0
        elif len(item_male[i]) == 0:
            extra = np.mean(item_female[i])
        elif len(item_female[i]) == 0:
            extra = np.mean(item_male[i])
        else:
            male_mean_rating_i = np.mean(item_male[i])
            female_mean_rating_i = np.mean(item_female[i])
            extra = male_mean_rating_i - female_mean_rating_i
        fairness_rethink_K += np.fabs(extra)

    fairness_rethink_K = fairness_rethink_K/item_num

    return fairness_rethink_K

def fairness_K_re_all(user_embedding,item_embedding,train_user_list,gender_data):
    '''
    compute rethinking_fairness@all
    '''
    user_e = user_embedding.cpu().detach().numpy()
    item_e = item_embedding.cpu().detach().numpy()
    all_prediction = np.matmul(user_e, item_e.T)


    for user in range(user_num):
        for i in train_user_list[user].keys():
            all_prediction[user][i] = 0

    # gender remove female rating; gender2 remove male rating
    # get 2 rating matrix
    gender_data_2 = np.array([1]) - gender_data
    gender_data = gender_data.reshape(-1,1)
    gender_data_2 = gender_data_2.reshape(-1, 1)
    # pdb.set_trace()
    male_matrix = all_prediction * gender_data
    female_matrix = all_prediction * gender_data_2

    # compute number of no zero
    # get 2 number matrix
    male_matrix2 = copy.deepcopy(male_matrix)
    female_matrix2 = copy.deepcopy(female_matrix)
    male_matrix2[male_matrix2!=0] = 1
    female_matrix2[female_matrix2!=0] =1

    fair_tmp=0
    for item in range(item_num):
        tmp_male = male_matrix[:,item]
        tmp_male_num = male_matrix2[:,item].sum()
        male_mean = tmp_male.sum()/tmp_male_num

        tmp_female = female_matrix[:, item]
        tmp_female_num = female_matrix2[:, item].sum()
        female_mean = tmp_female.sum() / tmp_female_num

        tmp_rating = np.fabs(male_mean-female_mean)
        fair_tmp += tmp_rating

    fair_all = fair_tmp/item_num
    return fair_all

def fairness_nips2017_1(test_pair,users_embedding,items_embedding,gender_data,item_size):
    '''
    compute nips 2017 beyond parity: U val
    '''

    test_set_len = len(test_pair)

    test_dataset2 = TripletUniformPair(test_pair, test_set_len)
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

    user_e = users_embedding.cpu().detach().numpy()
    item_e = items_embedding.cpu().detach().numpy()

    all_prediction = np.matmul(user_e, item_e.T)

    fairness_male_predict, fairness_female_predict = {},{}
    fairness_male_rating, fairness_female_rating = {},{}
    for i in range(item_size):
        fairness_male_predict[i] = []
        fairness_female_predict[i] = []
        fairness_male_rating[i] = []
        fairness_female_rating[i] = []
    '''
    for one item. get female predict,female rating; male predict,male rating.
    4 list, then compute mean value.
    '''

    for idx, (u, i, r) in enumerate(test_loader2):
        u=u.numpy()[0]
        i=i.numpy()[0]
        r=r.numpy()[0]

        prediction=all_prediction[u][i]
        rating = r
        if gender_data[u] == 0:
            fairness_male_predict[i].append(prediction)
            fairness_male_rating[i].append(rating)
        else:
            fairness_female_predict[i].append(prediction)
            fairness_female_rating[i].append(rating)

    value_list=[]
    for i in range(item_num):
        # if one item isn't rated by women or men, make the rating be 0.
        if len(fairness_male_predict[i]) !=0 and  len(fairness_female_predict[i]) != 0:
            value_female = np.mean(fairness_female_predict[i])-np.mean(fairness_female_rating[i])
            value_male = np.mean(fairness_male_predict[i])-np.mean(fairness_male_rating[i])
            values = np.fabs(value_female-value_male)
            value_list.append(values)
    # value_list: item_num size list. we need to compute its mean value. by /item_num or just np.mean()
    U_value = np.mean(np.array(value_list))

    return U_value

def nips2017_picked_item(test_pair,users_embedding,items_embedding,gender_data,item_size):
    '''
    compute nips 2017 beyond parity: U val
    to remove the information of no item, delete them.
    '''

    test_set_len = len(test_pair)

    test_dataset2 = TripletUniformPair(test_pair, test_set_len)
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

    user_e = users_embedding.cpu().detach().numpy()
    item_e = items_embedding.cpu().detach().numpy()

    all_prediction = np.matmul(user_e, item_e.T)

    fairness_male_predict, fairness_female_predict = {},{}
    fairness_male_rating, fairness_female_rating = {},{}
    for i in range(item_num):
    # for i in item_size:
        fairness_male_predict[i] = []
        fairness_female_predict[i] = []
        fairness_male_rating[i] = []
        fairness_female_rating[i] = []
    '''
    for one item. get female predict,female rating; male predict,male rating. 
    4 list, then compute mean value.
    '''

    for idx, (u, i, r) in enumerate(test_loader2):
        u=u.numpy()[0]
        i=i.numpy()[0]
        r=r.numpy()[0]

        prediction=all_prediction[u][i]

        rating = r
        if gender_data[u] == 0:
            fairness_male_predict[i].append(prediction)
            fairness_male_rating[i].append(rating)
        else:
            fairness_female_predict[i].append(prediction)
            fairness_female_rating[i].append(rating)

    value_list=[]
    for i in item_size:
        # if one item isn't rated by women or men, make the rating be 0.
        if len(fairness_male_predict[i]) !=0 and  len(fairness_female_predict[i]) != 0:
            value_female = np.fabs(np.mean(fairness_female_predict[i])-np.mean(fairness_female_rating[i]))
            value_male = np.fabs(np.mean(fairness_male_predict[i])-np.mean(fairness_male_rating[i]))
            values = np.fabs(value_female-value_male)
            value_list.append(values)

    # we only cam some items, not all.
    # print(len(value_list))
    U_value = np.mean(np.array(value_list))

    return U_value

def fairness_EO(test_pair,users_embedding,items_embedding,gender_data):
    '''
    design a new metrics about U val
    to remove the information of no item, delete them.
    '''
    test_pair = np.array(test_pair)
    gender_data2 = np.array([1]) - gender_data

    test_u = test_pair[:, 0].astype(int)
    test_i = test_pair[:, 1].astype(int)
    user_e = users_embedding.cpu().detach().numpy()
    item_e = items_embedding.cpu().detach().numpy()
    all_prediction = np.matmul(user_e, item_e.T)

    test_prediction = all_prediction[test_u,test_i]

    test_rating = test_pair[:,2]
    # test_bias = test_prediction-test_rating
    test_bias = np.fabs(test_prediction-test_rating)
    # pdb.set_trace()

    test_u = test_pair[:,0].astype(int).tolist()
    female_index= gender_data[test_u]
    male_index = gender_data2[test_u]

    female_num = female_index.sum()
    male_num = male_index.sum()

    female_bias = sum(test_bias*female_index)/female_num
    male_bias = sum(test_bias*male_index)/male_num

    U_value = female_bias - male_bias
    # pdb.set_trace()
    return U_value,female_bias,male_bias

def fairness_nips2017_3(test_pair,users_embedding,items_embedding,gender_data,item_size):
    '''
    compute nips 2017 beyond parity: U val
    to remove the information of no item, delete them.
    get out weights' information of item rated number.
    '''

    test_set_len = len(test_pair)

    test_dataset2 = TripletUniformPair(test_pair, test_set_len)
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

    user_e = users_embedding.cpu().detach().numpy()
    item_e = items_embedding.cpu().detach().numpy()

    all_prediction = np.matmul(user_e, item_e.T)

    fairness_male, fairness_female = {},{}
    for i in range(item_size):
        fairness_male[i] = []
        fairness_female[i] = []
    '''
    for one item. get female predict,female rating; male predict,male rating. 
    4 list, then compute mean value.
    '''

    for idx, (u, i, r) in enumerate(test_loader2):
        u=u.numpy()[0]
        i=i.numpy()[0]
        r=r.numpy()[0]

        prediction=all_prediction[u][i]
        rating = r
        if gender_data[u] == 0:
            fairness_male[i].append(np.fabs(prediction-rating))
        else:
            fairness_female[i].append(np.fabs(prediction-rating))

    value_list=[]
    for i in range(item_num):
        # if one item isn't rated by women or men, make the rating be 0.
        if len(fairness_male[i]) !=0 and  len(fairness_female[i]) != 0:
            values = np.fabs(np.mean(fairness_male[i])-np.mean(fairness_female[i]))
            value_list.append(values)
        elif len(fairness_male[i]) ==0 and  len(fairness_female[i]) != 0:
            values = np.mean(fairness_female[i])
            value_list.append(values)
        elif len(fairness_male[i]) !=0 and  len(fairness_female[i]) == 0:
            values = np.mean(fairness_male[i])
            value_list.append(values)
        else:
            continue


    # we only cam some items, not all.
    # print(len(value_list))
    U_value = np.mean(value_list)

    return U_value

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
# --------------------------load feature data-------------------------#
def load_item_kind():
    u_cols = ['item_id', 'name', 'cls']
    users = pd.read_csv('./ml-1m/movies.dat', sep='::', names=u_cols,
                        encoding='latin-1', parse_dates=True,engine='python')
    # pdb.set_trace()
    # # users['item_id'] = users['item_id']-1
    # pdb.set_trace()
    users1 = users[users.cls.str.contains('Romance')]
    users2 = users[users.cls.str.contains('Action')]
    users3 = users[users.cls.str.contains('Crime')]
    users4 = users[users.cls.str.contains('Musical')]
    users5 = users[users.cls.str.contains('Sci-Fi')]
    # pdb.set_trace()

    item_ids = set(users1['item_id']) | set(users2['item_id']) | set(users3['item_id']) | set(users4['item_id']) |set(users5['item_id'])
    # pdb.set_trace()
    return item_ids

def load_item_new_ids():
    s = MovieLens1M('./ml-1m')
    df = s.load()
    df['user'] = df['user'] - 1
    df, itemid = convert_unique_idx(df, 'item')
    item_old_ids = load_item_kind()
    item_new_ids = []
    for i in item_old_ids:
        if i in itemid.keys():
            item_new_ids.append(itemid[i])
    return item_new_ids