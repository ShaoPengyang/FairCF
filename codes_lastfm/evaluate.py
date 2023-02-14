import torch
import torch.nn as nn
import os
import numpy as np
import random
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
user_num = 359347
item_num = 292589

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
    y_test = np.asarray(y_test).squeeze()
    y_pred= np.asarray(y_pred).squeeze()
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_pred, average=average)

def metrics(preds_list,labels_list,probs_list,dataset_len,correct):
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

# def fairness_nips2017_1(test_pair,items_num,users_embedding,items_embedding,users_bias,items_bias,gender_data,avg_train_mean):
#     '''
#     compute nips 2017 beyond parity: U val
#     '''
#
#     items_save = np.zeros((item_num,2))
#     # items_save to save female sum  and  male sum
#
#     for u,i,r in test_pair:
#         # a=time.time()
#         predict = np.matmul(users_embedding[u], items_embedding[i].T) + avg_train_mean +users_bias[u] + items_bias[i]
#         bias = predict - r
#         items_save[i][gender_data[u]] += bias
#         # print(time.time()-a)
#
#     all_bias = items_save * items_num
#     all_bias_del = all_bias[:,0]-all_bias[:,1]
#     # pdb.set_trace()
#     all_bias_del = np.fabs(all_bias_del)
#     return np.mean(all_bias_del)

def fair_K_lastfm_all(user_e,item_e,gender_data,train_user_list):
    '''
    compute rethinking_fairness@ALL among random picked 10000 items
    step1: for every user, get their rating and divide due to gender
    step2: for every item, cal rating
    '''
    begin_eval=time.time()
    gender_data = np.array(gender_data)
    # pdb.set_trace()
    all_prediction = np.matmul(user_e, item_e.T)

    male_rating = all_prediction*(gender_data.reshape(-1,1))
    female_rating = all_prediction*((1-gender_data).reshape(-1,1))

    male_rating = male_rating.sum(0)/(gender_data.sum())
    female_rating = female_rating.sum(0)/((1-gender_data).sum())
    fairness_rethink_K_all = (np.fabs(male_rating-female_rating)).mean()
    # pdb.set_trace()
    topk = 50
    set_list_all = set(range(2500))
    item_female = defaultdict(list)
    item_male = defaultdict(list)
    for userid in range(user_e.shape[0]):
        item_i_list = list(set_list_all - set(train_user_list[userid].keys()))
        prediction = all_prediction[userid][item_i_list]
        indices1 = largest_indices(prediction, topk)
        # pdb.set_trace()
        indices1 = list(indices1[0])
        item_i_list = np.array(item_i_list)
        top_k_index = item_i_list[indices1]
        # pdb.set_trace()
        for i in top_k_index:
            if gender_data[userid] == 0:
                item_male[i].append(all_prediction[userid][i])
            else:
                item_female[i].append(all_prediction[userid][i])
    fairness_rethink_K2 = 0
    length = item_e.shape[0]
    for i in range(item_e.shape[0]):
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
            # pdb.set_trace()
        fairness_rethink_K2 += np.fabs(extra)
    fairness_rethink_K2 = fairness_rethink_K2 / length
    print(length)
    print("fairness@all end time:\t" + str(round((time.time() - begin_eval), 4)))
    return fairness_rethink_K_all,fairness_rethink_K2
    # pdb.set_trace()
    # for userid in range(user_e.shape[0]):
    #     item_i_list = np.array(item_i_list)
    #     top_k_index = item_i_list[indices1]
    #     for i in top_k_index:
    #         if gender_data[userid] == 0:
    #             item_male[i].append(all_prediction[userid][i])
    #         else:
    #             item_female[i].append(all_prediction[userid][i])
    # fairness_rethink_K = 0
    # for i in range(item_e.shape[0]):
    #     if len(item_male[i]) == 0 and len(item_female[i]) == 0:
    #         extra = 0
    #     elif len(item_male[i]) == 0:
    #         extra = np.mean(item_female[i])
    #     elif len(item_female[i]) == 0:
    #         extra = np.mean(item_male[i])
    #     else:
    #         male_mean_rating_i = np.mean(item_male[i])
    #         female_mean_rating_i = np.mean(item_female[i])
    #         extra = male_mean_rating_i - female_mean_rating_i
    #     fairness_rethink_K += np.fabs(extra)
    # fairness_rethink_K = fairness_rethink_K / 10000
    # print("fairness@all end time:\t" + str(round((time.time() - begin_eval), 4)))
    # return fairness_rethink_K
