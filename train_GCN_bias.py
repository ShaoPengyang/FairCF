#-- coding:UTF-8 --
'''
train RMSE with ml-1m     with compositional adversary 2020/3/27
'''

import torch
import torch.nn as nn
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
from torch.nn.utils import clip_grad_norm
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import torch.autograd as autograd
import argparse
import pdb
import pickle
from collections import defaultdict
import time
import copy
from tqdm import tqdm
tqdm.monitor_interval = 0
from models import *
from evaluate import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="change gpuid")
parser.add_argument("-g", "--gpu-id", help="choose which gpu to use", type=str, default=str(3))
parser.add_argument("-t", "--times", help="choose times", type=str, default='8')
# g 5 fair cf age
# g 6 fair cf com
# g 7 icml age
# g 8 icml com
parser.add_argument("-d", "--D_steps", help="num of train Disc times", type=int, default=2)
# 0 epoch 41 no adversary
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
D_steps = args.D_steps
run_id = 'g' + args.times
dataset_base_path = './ml-100k'
data_dir = './ml-100k'
dataset_name = 'ml-100k'
path_save_model_base = './model/' + dataset_name + '/' + run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

path_save_log_base = './Logs/' + dataset_name + '/' + run_id

if (os.path.exists(path_save_log_base)):
    print('has log save path')
else:
    os.makedirs(path_save_log_base)
user_num = 6040
item_num = 3706
factor_num = 64
batch_size = 8192*4
# old 8192*4
lamada = 0.01

def readTrainSparseMatrix(dict_matrix,is_user,u_d,i_d,size):
    '''
    这里要修改成，每次输入评分 1-5 的时候
    :param dict_matrix: [dict1,dict2,dict3] len(user/item size)
    :param is_user:
    :return:
    '''
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d # save  1/ len(item[user])+1
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in range(size):
        len_set=len(dict_matrix[i])
        for j,r in dict_matrix[i].items():
            # if r == ratings:
            user_items_matrix_i.append([i,j]) # i userid  j item id
            # 可以理解为计算popularity
            # di_i user i 对应的 1/ len(item[user])+1 dj_j item i 对应的
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            #1/sqrt((d_i+1)(d_j+1))0
            user_items_matrix_v.append(d_i_j)

    # pdb.set_trace()
    user_items_matrix_i = torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v = torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

def readD(user_list,size):
    user_D = []
    for user in range(size):
        length = 0
        for item,r in user_list[user].items():
            # if r == rating:
            length += 1
        len_dict = 1.0/(length+1)
        user_D.append(len_dict)
    return user_D

def baseline_gcn():
    print('------------training processing--------------')
    with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        all_user_list = dataset['all_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
    print('Load complete')

    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list,item_size)
    sparse_u_i=readTrainSparseMatrix(train_user_list, True, user_d, item_d,user_size)
    sparse_i_u=readTrainSparseMatrix(train_item_list, False,user_d,item_d,item_size)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g0'
    path_save_model = './model/' + dataset_name + '/g0'
    result_file = open(path_save_log_base + '/1206_bias_gcn.txt', 'a')
    result_file.write('\n\n\n log gcn rating ==  \n')

    model = GCN_bias(user_size, item_size, factor_num,sparse_u_i,sparse_i_u,user_d,item_d,avg_rating).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            u = u.cuda()
            i = i.cuda()
            rating = r.cuda()
            optimizer.zero_grad()
            task_loss, loss2, l2_regulization = model(u, i, rating)
            task_loss.backward()
            optimizer.step()
            train_loss_sum.append(task_loss.item())

        PATH_model = path_save_model + '/GCN_bias_' + str(epoch) + '(2).pt'
        torch.save(model.state_dict(), PATH_model)

        model.eval()
        with torch.no_grad():
            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            fairness_all, fairness_K50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                              train_user_list)
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()

            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse  = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()

        elapsed_time = time.time() - start_time
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss)
        print('--train_time--', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 6)) + '\t test rmse:' + str(round(test_rmse, 6)) + '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def baseline_faircf():
    print('------------training processing--------------')
    # with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
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
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    # item_inds1 = np.array(item_inds)
    item_inds = np.load('preprocessed/item_labels.npy')
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list,item_size)
    sparse_u_i=readTrainSparseMatrix(train_user_list, True, user_d, item_d,user_size)
    sparse_i_u=readTrainSparseMatrix(train_item_list, False,user_d,item_d,item_size)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g0'
    path_save_model = './model/' + dataset_name + '/g0'
    result_file = open(path_save_log_base + '/rebuttal_faircf.txt', 'a')

    model = GCN_bias_faircf(user_size, item_size, factor_num,sparse_u_i,sparse_i_u,user_d,item_d,avg_rating).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    regression_gender_item = GenderRegression_onedim(factor_num, item_inds).cuda()
    optimizer_regression_gender_item = torch.optim.Adam(regression_gender_item.parameters(), lr=0.005)
    ################
    #   pretrain   #
    ################
    model.train()
    fairD_gender_user.train()
    regression_gender_item.train()
    for epoch in range(15):
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss,loss2,d_loss,l2_loss = model(u, i, r)
            task_loss.backward()
            optimizer.step()

        with torch.no_grad():
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
        print(val_rmse)
    torch.save(model.state_dict(), './preprocessed/recommend_pretrain_gcn_rebuttal.pt')

    # pdb.set_trace()
    fairD_gender_user.train()
    for epoch in range(20):
        for idx, (u, i, r) in enumerate(train_loader):
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r,return_batch_embedding=True)
            fairD_gender_user.zero_grad()
            l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
            l_penalty_user.backward()
            optimizer_fairD_gender_user.step()
    torch.save(fairD_gender_user.state_dict(), './preprocessed/DISC_pretrain_gcn_rebuttal.pt')

    regression_gender_item.train()
    for epoch in range(20):
        for idx, (u, i, r) in enumerate(train_loader):
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
            regression_gender_item.zero_grad()
            l_penalty_user = regression_gender_item(item_embedding.detach(), i, True)
            l_penalty_user.backward()
            optimizer_regression_gender_item.step()
    torch.save(regression_gender_item.state_dict(), './preprocessed/REG_pretrain_gcn_rebuttal.pt')
    pdb.set_trace()
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_gcn_rebuttal.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_gcn_rebuttal.pt"))
    regression_gender_item.load_state_dict(torch.load("./preprocessed/REG_pretrain_gcn_rebuttal.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum4 = []
        train_loss_sum5 = []
        train_loss_sum6 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            freeze_model(regression_gender_item)
            fairD_gender_user.eval()
            regression_gender_item.eval()
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss, loss2, d_loss, r_loss = model(u, i, r, Disc=fairD_gender_user,
                                                     Reg=regression_gender_item)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())
            train_loss_sum4.append(r_loss.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user,
                                                       Reg=regression_gender_item)

            unfreeze_model(fairD_gender_user)
            unfreeze_model(regression_gender_item)
            D_steps = 20
            for _ in range(10):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())

            for _ in range(10):
                regression_gender_item.train()
                regression_gender_item.zero_grad()
                l_penalty_item = regression_gender_item(item_embedding.detach(), i, True)
                l_penalty_item.backward()
                optimizer_regression_gender_item.step()
                if _ == 0:
                    train_loss_sum6.append(l_penalty_item.item())

            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(y.view_as(preds)).sum().item()
                preds_list4.append(preds)
                probs_list4.append(y_hat)
                labels_list4.append(y)
                age_correct.append(correct)
                AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)
                AUC_.append(AUC)
                acc_.append(acc)

        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g0'
        PATH_model = path_save_model_b + '/rebuttal_' + str(epoch) + '.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        # training finish, we want to cum fairness and rmse
        with torch.no_grad():
            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            fairness_all, fairness_K50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                              train_user_list, 50)
            # fairness_all2 = fairness_K_re_all__(user_e, item_e, user_bias_tmp, item_bias_tmp, train_user_list,
            #                                     gender_data)
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss4 = round(np.mean(train_loss_sum4[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        train_loss6 = round(np.mean(train_loss_sum6[:-1]), 4)
        AUC_mean = round(np.mean(AUC_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t r loss1:" + str(
            train_loss4) + "\t r loss2:" + str(train_loss6) + "\t acc:" + str(acc_mean) + "\t AUC:" + str(AUC_mean)

        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4)) + '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(
            round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def baseline_faircf_user():
    print('------------training processing--------------')
    with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
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
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    item_inds = np.array(item_inds)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list,item_size)
    sparse_u_i=readTrainSparseMatrix(train_user_list, True, user_d, item_d,user_size)
    sparse_i_u=readTrainSparseMatrix(train_item_list, False,user_d,item_d,item_size)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g0'
    path_save_model = './model/' + dataset_name + '/g0'
    result_file = open(path_save_log_base + '/1206_bias_gcn_faircf_user.txt', 'a')

    model = GCN_bias_faircf_user(user_size, item_size, factor_num,sparse_u_i,sparse_i_u,user_d,item_d,avg_rating).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    ################
    #   pretrain   #
    ################
    model.train()
    fairD_gender_user.train()

    # for epoch in range(30):
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         model.train()
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         model.zero_grad()
    #         task_loss,loss2,d_loss,l2_loss = model(u, i, r)
    #         task_loss.backward()
    #         optimizer.step()
    #
    #     with torch.no_grad():
    #         for idx, (u, i, r) in enumerate(val_loader):
    #             u, i, r = u.cuda(), i.cuda(), r.cuda()
    #             rmse = model.predict(u, i, r)
    #             val_rmse = torch.sqrt(rmse)
    #         val_rmse = val_rmse.item()
    #     print(val_rmse)
    # torch.save(model.state_dict(), './preprocessed/recommend_pretrain_gcn.pt')
    #
    # # pdb.set_trace()
    # fairD_gender_user.train()
    # for epoch in range(20):
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r,return_batch_embedding=True)
    #         fairD_gender_user.zero_grad()
    #         l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
    #         l_penalty_user.backward()
    #         optimizer_fairD_gender_user.step()
    # torch.save(fairD_gender_user.state_dict(), './preprocessed/DISC_pretrain_gcn.pt')
    #
    # regression_gender_item.train()
    # for epoch in range(20):
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
    #         regression_gender_item.zero_grad()
    #         l_penalty_user = regression_gender_item(item_embedding.detach(), i, True)
    #         l_penalty_user.backward()
    #         optimizer_regression_gender_item.step()
    # torch.save(regression_gender_item.state_dict(), './preprocessed/REG_pretrain_gcn.pt')
    # pdb.set_trace()
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_gcn.pt"))
    # model.user_bias.weight.requires_grad = False
    # model.user_bias.weight.requires_grad = False
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_gcn.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum5 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            fairD_gender_user.eval()

            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss, loss2, d_loss, r_loss = model(u, i, r, Disc=fairD_gender_user)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)

            unfreeze_model(fairD_gender_user)
            for _ in range(10):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())


            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(y.view_as(preds)).sum().item()
                preds_list4.append(preds)
                probs_list4.append(y_hat)
                labels_list4.append(y)
                age_correct.append(correct)
                AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)
                AUC_.append(AUC)
                acc_.append(acc)

        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g0'
        PATH_model = path_save_model_b + '/cfu_' + str(epoch) + '.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        # training finish, we want to cum fairness and rmse
        with torch.no_grad():
            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            fairness_all, fairness_K50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                              train_user_list, 50)
            # fairness_all2 = fairness_K_re_all__(user_e, item_e, user_bias_tmp, item_bias_tmp, train_user_list,
            #                                     gender_data)
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        AUC_mean = round(np.mean(AUC_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t acc:" + str(acc_mean) + "\t AUC:" + str(AUC_mean)

        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4)) + '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(
            round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def baseline_icml_2019():
    print('------------training processing--------------')
    with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
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
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    item_inds = np.array(item_inds)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list,item_size)
    sparse_u_i=readTrainSparseMatrix(train_user_list, True, user_d, item_d,user_size)
    sparse_i_u=readTrainSparseMatrix(train_item_list, False,user_d,item_d,item_size)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g0'
    path_save_model = './model/' + dataset_name + '/g0'
    result_file = open(path_save_log_base + '/1206_bias_gcn_icml_2019.txt', 'a')

    model = GCN_icml_2019(user_size, item_size, factor_num,sparse_u_i,sparse_i_u,user_d,item_d,avg_rating).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    ################
    #   pretrain   #
    ################
    model.train()
    fairD_gender_user.train()
    # for epoch in range(8):
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         model.train()
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         model.zero_grad()
    #         task_loss,loss2,d_loss,l2_loss = model(u, i, r)
    #         task_loss.backward()
    #         optimizer.step()
    #
    #     with torch.no_grad():
    #         for idx, (u, i, r) in enumerate(val_loader):
    #             u, i, r = u.cuda(), i.cuda(), r.cuda()
    #             rmse = model.predict(u, i, r)
    #             val_rmse = torch.sqrt(rmse)
    #         val_rmse = val_rmse.item()
    #     print(val_rmse)
    # torch.save(model.state_dict(), './preprocessed/recommend_pretrain_gcn_icml.pt')
    #
    # # pdb.set_trace()
    # fairD_gender_user.train()
    # for epoch in range(20):
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r,return_batch_embedding=True)
    #         fairD_gender_user.zero_grad()
    #         l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
    #         l_penalty_user.backward()
    #         optimizer_fairD_gender_user.step()
    # torch.save(fairD_gender_user.state_dict(), './preprocessed/DISC_pretrain_gcn_icml.pt')
    #
    # pdb.set_trace()
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_gcn_icml.pt"))
    # model.user_bias.weight.requires_grad = False
    # model.user_bias.weight.requires_grad = False
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_gcn_icml.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum5 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            fairD_gender_user.eval()

            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss, loss2, d_loss, r_loss = model(u, i, r, Disc=fairD_gender_user)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)

            unfreeze_model(fairD_gender_user)
            for _ in range(10):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())


            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(y.view_as(preds)).sum().item()
                preds_list4.append(preds)
                probs_list4.append(y_hat)
                labels_list4.append(y)
                age_correct.append(correct)
                AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)
                AUC_.append(AUC)
                acc_.append(acc)

        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g0'
        PATH_model = path_save_model_b + '/icml_' + str(epoch) + '.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        # training finish, we want to cum fairness and rmse
        with torch.no_grad():
            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            fairness_all, fairness_K50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                              train_user_list, 50)
            # fairness_all2 = fairness_K_re_all__(user_e, item_e, user_bias_tmp, item_bias_tmp, train_user_list,
            #                                     gender_data)
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        AUC_mean = round(np.mean(AUC_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t acc:" + str(acc_mean) + "\t AUC:" + str(AUC_mean)

        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4)) + '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(
            round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def test_new_cls_and_reg():
    print('------------test embeds_cls_accuracy processing--------------')
    # with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
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
    print('Load complete')
    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list, item_size)
    sparse_u_i = readTrainSparseMatrix(train_user_list, True, user_d, item_d, user_size)
    sparse_i_u = readTrainSparseMatrix(train_item_list, False, user_d, item_d, item_size)

    # model = GCN_bias(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/GCN_bias_195(2).pt'

    model = GCN_bias_faircf(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/2adv_42.pt'
    Path_model = './model/ml-100k/g0/rebuttal_38.pt'

    # model = GCN_bias_faircf_user(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d,
    #                              avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/cfu_124.pt'
    #
    # model = GCN_icml_2019(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/icml_34.pt'

    model.train()
    model.load_state_dict(torch.load(Path_model))
    model.eval()
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)

    users = np.arange(user_size)
    np.random.seed(0)
    np.random.shuffle(users)
    cut = int(user_size * 0.8)
    train_users = users[:cut]
    test_users = users[cut:]
    train_users_index = np.array(train_users)
    test_users_index = np.array(test_users)
    train_len = len(train_users_index)
    test_len = len(test_users_index)

    model.train()
    with torch.no_grad():
        user_6040_embedding, item_3706_embedding, user_bias, item_bias = model.predict(0, 0, 0, return_e=True)
    all_user_embedding = user_6040_embedding.detach()
    model.eval()
    train_set_len, test_set_len, val_set_len = len(train_pair), len(test_pair), len(val_pair)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    for idx, (u, i, r) in enumerate(val_loader):
        u, i, r = u.cuda(), i.cuda(), r.cuda()
        rmse = model.predict(u, i, r)
        val_rmse = torch.sqrt(rmse)
    val_rmse = val_rmse.item()

    for epoch in tqdm(range(120)):
        fairD_gender_user.train()
        p_batch = torch.cuda.LongTensor(train_users_index)
        filter_l_emb= all_user_embedding[p_batch]
        l_penalty_2 = fairD_gender_user(filter_l_emb.detach(), p_batch, return_loss=True)
        optimizer_fairD_gender_user.zero_grad()
        l_penalty_2.backward()
        optimizer_fairD_gender_user.step()
        train_loss_sum = l_penalty_2.item()
        # pdb.set_trace()

        fairD_gender_user.eval()
        with torch.no_grad():
            dataset_len = train_len
            preds_list4, labels_list4, probs_list4 = [], [], []
            y_hat, y = fairD_gender_user.predict(filter_l_emb.detach(), p_batch,return_loss=False, return_preds=False)
            preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            correct = preds.eq(y.view_as(preds)).sum().item()
            preds_list4.append(preds)
            probs_list4.append(y_hat)
            labels_list4.append(y)
            AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)

            # testing set's correct
            p_batch_2 = torch.cuda.LongTensor(test_users_index)
            filter_l_emb_2 = all_user_embedding[p_batch_2]
            dataset_len = test_len
            preds_list, labels_list, probs_list = [], [], []
            y_hat_2, l_A_labels = fairD_gender_user.predict(filter_l_emb_2.detach(), p_batch_2,return_loss=False, return_preds=False)
            test_loss = fairD_gender_user.predict(filter_l_emb_2.detach(), p_batch_2, return_loss=True)
            test_loss = test_loss.item()

            preds = (y_hat_2 > torch.Tensor([0.5]).cuda()).float() * 1
            l_correct = preds.eq(l_A_labels.view_as(preds)).sum().item()
            preds_list.append(preds)
            probs_list.append(y_hat_2)
            labels_list.append(l_A_labels)
            AUC1, acc1, f1, f1_macro1 = metrics(preds_list, labels_list, probs_list, dataset_len, l_correct)

            str_print_test = "Train auc is: %f Test auc is: %f loss is: %f test loss: %f test acc: %f" % (AUC,AUC1,train_loss_sum,test_loss,acc1)
            print(str_print_test)
    print(val_rmse)

    # items = np.arange(item_size)
    # np.random.shuffle(items)
    # cut = int(item_size * 0.8)
    # train_items = items[:cut]
    # test_items = items[cut:]
    # train_items_index = np.array(train_items)
    # test_items_index = np.array(test_items)
    # train_len = len(train_items_index)
    # test_len = len(test_items_index)
    # train_set = KBDataset(train_users_index)
    # print("len(train_set):" + str(len(train_set)))
    # test_set = KBDataset(test_users_index)
    # print("len(test_set):" + str(len(test_set)))
    # for epoch in range(200):
    #     regression_gender_item.train()
    #     '''
    #     这里的评价指标打算写成 rmse，观察reg的rmse，mse loss，一次性读取数据，可以复制粘贴其他地方的代码
    #     '''
    #     p_batch = torch.cuda.LongTensor(train_items_index)
    #
    #     filter_l_emb = all_item_embedding[p_batch]
    #     l_penalty_2 = regression_gender_item(filter_l_emb.detach(), p_batch, return_loss=True)
    #     optimizer_regression_gender_item.zero_grad()
    #     l_penalty_2.backward()
    #     optimizer_regression_gender_item.step()
    #
    #     regression_gender_item.eval()
    #     with torch.no_grad():
    #         # training set's correct
    #         train_rmse = regression_gender_item.predict(filter_l_emb.detach(), p_batch)
    #         rmse1 = torch.sqrt(train_rmse).item()
    #         p_batch_2 = torch.cuda.LongTensor(test_items_index)
    #
    #         filter_l_emb_2 = all_item_embedding[p_batch_2]
    #         test_rmse = regression_gender_item.predict(filter_l_emb_2.detach(),p_batch_2)
    #         rmse2 = torch.sqrt(test_rmse).item()
    #
    #     print("train rmse:"+str(rmse1)+"\t test rmse:"+str(rmse2))

def fair_K_lastfm_bias__(user_e,item_e,user_bias,item_bias,gender_data,train_user_list,topk):
    '''
    compute rethinking_fairness@ALL among random picked 10000 items
    step1: for every user, get their rating and divide due to gender
    step2: for every item, cal rating
    '''
    begin_eval=time.time()
    gender_data = np.array(gender_data)
    # pdb.set_trace()
    all_prediction = np.matmul(user_e, item_e.T) + user_bias + item_bias

    set_list_all = set(range(item_num))
    item_female = defaultdict(list)
    item_male = defaultdict(list)
    item_female_all = defaultdict(list)
    item_male_all = defaultdict(list)
    for userid in range(user_num):
        item_i_list = list(set_list_all - set(train_user_list[userid].keys()))
        for item_id in item_i_list:
            if gender_data[userid] == 0:
                item_male_all[item_id].append(all_prediction[userid][item_id])
            else:
                item_female_all[item_id].append(all_prediction[userid][item_id])
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
    fairness_rethink_K_all = 0
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

    extra = 0
    for i in range(item_e.shape[0]):
        if len(item_male_all[i]) == 0 and len(item_female_all[i]) == 0:
            extra = 0
        elif len(item_male_all[i]) == 0:
            extra = np.mean(item_female_all[i])
        elif len(item_female_all[i]) == 0:
            extra = np.mean(item_male_all[i])
        else:
            male_mean_rating_i = np.mean(item_male_all[i])
            female_mean_rating_i = np.mean(item_female_all[i])
            extra = male_mean_rating_i - female_mean_rating_i
            # pdb.set_trace()
        fairness_rethink_K_all += np.fabs(extra)
    fairness_rethink_K_all = fairness_rethink_K_all / length

    print("fairness@all end time:\t" + str(round((time.time() - begin_eval), 4)))
    return fairness_rethink_K_all,fairness_rethink_K2

def fair_K_lastfm_bias__2(user_e,item_e,user_bias,item_bias,gender_data,train_user_list,topk):
    '''
    compute rethinking_fairness@ALL among random picked 10000 items
    step1: for every user, get their rating and divide due to gender
    step2: for every item, cal rating
    '''
    gender_data = np.array(gender_data)
    all_prediction = np.matmul(user_e, item_e.T) + user_bias + item_bias
    set_list_all = set(range(item_num))
    item_female = defaultdict(list)
    item_male = defaultdict(list)
    for userid in range(user_num):
        item_i_list = list(set_list_all - set(train_user_list[userid].keys()))
        prediction = all_prediction[userid][item_i_list]
        indices1 = largest_indices(prediction, topk)
        # pdb.set_trace()
        indices1 = list(indices1[0])
        item_i_list = np.array(item_i_list)
        top_k_index = item_i_list[indices1]
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
        fairness_rethink_K2 += np.fabs(extra)
    fairness_rethink_K2 = fairness_rethink_K2 / length

    return fairness_rethink_K2

def test_MF():
    print('------------training processing--------------')
    with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
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
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    item_inds = np.array(item_inds)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list, item_size)
    sparse_u_i = readTrainSparseMatrix(train_user_list, True, user_d, item_d, user_size)
    sparse_i_u = readTrainSparseMatrix(train_item_list, False, user_d, item_d, item_size)

    train_set_len, test_set_len, val_set_len = len(train_pair), len(test_pair), len(val_pair)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)

    model = GCN_bias_faircf(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    Path_model = './model/ml-1m/g0/2adv_42.pt'
    # Path_model = './model/ml-1m/g0/w200_45.pt'

    # model = GCN_bias_faircf_user(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/cfu_124.pt'

    # model = GCN_icml_2019(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/icml_34.pt'

    # model = GCN_bias(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/GCN_bias_36(2).pt'

    print(Path_model)
    model.train()
    model.load_state_dict(torch.load(Path_model))
    model.eval()
    test_rmse = 0
    with torch.no_grad():
        for idx, (u, i, r) in enumerate(test_loader):
            # print(idx)
            u, i, r = u.cuda(), i.cuda(), r.cuda()
            rmse = model.predict(u, i, r)
            test_rmse = torch.sqrt(rmse)
        test_rmse = test_rmse.item()
        print(test_rmse)
        users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
    user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
    item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
    user_bias = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
    item_bias = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
    # fairness_All = new_fairness_all(user_e, item_e, user_bias, item_bias, gender_data)
    # print("\t fair@50:" + str(fairness_All))
    user_bias_tmp = user_bias
    item_bias_tmp = item_bias
    gender_data = gender_data
    fairness_all, fairness50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                  train_user_list, 50)

    fairness10 = fair_K_lastfm_bias__2(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 10)
    fairness20 = fair_K_lastfm_bias__2(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 20)
    fairness30 = fair_K_lastfm_bias__2(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 30)
    fairness40 = fair_K_lastfm_bias__2(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 40)
    rmse_str = "\t test rmse:" + str(test_rmse)
    nips_str = "\t fair@10:" + str(fairness10) + "\t fair@20:" + str(fairness20) + "\t fair@30:" + str(
        fairness30)  + "\t fair@40:" + str(fairness40) + "\t fair@50:" + str(fairness50) + "\t fair@all:" + str(fairness_all)
    print(rmse_str)
    print(nips_str)

def baseline_fair_age():
    print("fair age")
    with open('./preprocessed/ml-1m_gcn_re2.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        # item_inds = dataset['item_inds']
        # # age_data = dataset['age_data']
        item_age_inds = dataset['item_age_inds']
        age_labels = dataset['age_labels']
    age_labels = np.array(age_labels)
    item_age_inds = np.array(item_age_inds)
    # pdb.set_trace()
    print('Load complete')

    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating
    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    # item_inds1 = np.array(item_inds)
    item_inds = np.load('preprocessed/item_labels.npy')
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list, item_size)
    sparse_u_i = readTrainSparseMatrix(train_user_list, True, user_d, item_d, user_size)
    sparse_i_u = readTrainSparseMatrix(train_item_list, False, user_d, item_d, item_size)

    train_set_len, test_set_len, val_set_len = len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g5'
    result_file = open(path_save_log_base + '/fair_age.txt', 'a')  # ('./log/results_gcmc.txt','w+')
    result_file.write('Fair CF (Sensitive attribute is age)\n')

    model = GCN_bias_faircf(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = AgeDiscriminator(factor_num, age_labels).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)

    regression_gender_item = AgerRegression(factor_num, item_age_inds).cuda()
    optimizer_regression_gender_item = torch.optim.Adam(regression_gender_item.parameters(), lr=0.005)
    # ###############
    #   pretrain   #
    # ###############
    model.train()
    fairD_gender_user.train()
    regression_gender_item.train()
    model.load_state_dict(torch.load("./preprocessed/GCN_re2.pt"))

    print('model D')
    for epoch in range(40):
        loss = []
        for idx, (u, i, r) in enumerate(train_loader):
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
            fairD_gender_user.zero_grad()
            l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
            l_penalty_user.backward()
            loss.append(l_penalty_user.item())
            optimizer_fairD_gender_user.step()
        print(np.mean(loss[:-1]))
    torch.save(fairD_gender_user.state_dict(), './preprocessed/GCN_re2_AD.pt')
    regression_gender_item.train()
    print('model R')
    for epoch in range(1):
        loss = []
        for idx, (u, i, r) in enumerate(train_loader):
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
            regression_gender_item.zero_grad()
            l_penalty_user = regression_gender_item(item_embedding.detach(), i, True)
            l_penalty_user.backward()
            optimizer_regression_gender_item.step()
            loss.append(l_penalty_user.item())
        print(np.mean(loss[:-1]))
    torch.save(regression_gender_item.state_dict(), './preprocessed/GCN_re2_AR.pt')
    pdb.set_trace()



    model.load_state_dict(torch.load("./preprocessed/GCN_re2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/GCN_re2_AD.pt"))
    regression_gender_item.load_state_dict(torch.load("./preprocessed/GCN_re2_AR.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum4 = []
        train_loss_sum5 = []
        train_loss_sum6 = []
        f1_micro_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            freeze_model(regression_gender_item)
            fairD_gender_user.eval()
            regression_gender_item.eval()
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss, loss2, d_loss, r_loss = model(u, i, r, Disc=fairD_gender_user,
                                                     Reg=regression_gender_item)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())
            train_loss_sum4.append(r_loss.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user,
                                                       Reg=regression_gender_item)

            unfreeze_model(fairD_gender_user)
            unfreeze_model(regression_gender_item)
            for _ in range(40):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())

            for _ in range(40):
                regression_gender_item.train()
                regression_gender_item.zero_grad()
                l_penalty_item = regression_gender_item(item_embedding.detach(), i, True)
                l_penalty_item.backward()
                optimizer_regression_gender_item.step()
                if _ == 0:
                    train_loss_sum6.append(l_penalty_item.item())

            with torch.no_grad():
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user,
                                                       Reg=regression_gender_item)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                y_hat, y = y_hat.data.cpu().numpy(), y.data.cpu().numpy()
                preds = np.argmax(y_hat,axis=-1)
                correct = np.sum(preds==y)
                age_correct.append(correct)
                AUC, acc, f1_micro, f1_macro = metrics(preds, y, y_hat, dataset_len, correct)
                f1_micro_.append(f1_micro)
                acc_.append(acc)

        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g5'
        PATH_model = path_save_model_b + '/fair' + str(epoch) + '_re2.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        # training finish, we want to cum fairness and rmse
        with torch.no_grad():
            # users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            # user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            # item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            # user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            # item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            # fairness_all, fairness_K50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
            #                                                   train_user_list)
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss4 = round(np.mean(train_loss_sum4[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        train_loss6 = round(np.mean(train_loss_sum6[:-1]), 4)
        f1_micro_mean = round(np.mean(f1_micro_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t r loss1:" + str(
            train_loss4) + "\t r loss2:" + str(train_loss6) + "\t acc:" + str(acc_mean) + "\t f1:" + str(f1_micro_mean)

        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4)) #+ '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(
            #round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def baseline_fair_com():
    print("fair com")
    with open('./preprocessed/ml-1m_gcn_re2.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
        # # age_data = dataset['age_data']
        item_age_inds = dataset['item_age_inds']
        age_labels = dataset['age_labels']
    age_labels = np.array(age_labels)
    item_age_inds = np.array(item_age_inds)
    # pdb.set_trace()
    print('Load complete')

    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating
    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    item_inds = np.array(item_inds)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list, item_size)
    sparse_u_i = readTrainSparseMatrix(train_user_list, True, user_d, item_d, user_size)
    sparse_i_u = readTrainSparseMatrix(train_item_list, False, user_d, item_d, item_size)

    train_set_len, test_set_len, val_set_len = len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g5'
    result_file = open(path_save_log_base + '/fair_age.txt', 'a')  # ('./log/results_gcmc.txt','w+')
    result_file.write('Fair CF (Sensitive attribute is age)\n')

    model = GCN_compostional(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    regression_gender_item = GenderRegression_onedim(factor_num, item_inds).cuda()
    optimizer_regression_gender_item = torch.optim.Adam(regression_gender_item.parameters(), lr=0.005)

    fairD_age_user = AgeDiscriminator(factor_num, age_labels).cuda()
    optimizer_fairD_age_user = torch.optim.Adam(fairD_age_user.parameters(), lr=0.005)
    regression_age_item = AgerRegression(factor_num, item_age_inds).cuda()
    optimizer_regression_age_item = torch.optim.Adam(regression_age_item.parameters(), lr=0.005)
    # ###############
    #   pretrain   #
    # ###############
    model.train()
    model.load_state_dict(torch.load("./preprocessed/GCN_re2.pt"))
    fairD_gender_user.train()
    regression_gender_item.train()
    fairD_age_user.train()
    regression_age_item.train()
    # print('model D')
    # for epoch in range(40):
    #     loss = []
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
    #         fairD_gender_user.zero_grad()
    #         l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
    #         l_penalty_user.backward()
    #         loss.append(l_penalty_user.item())
    #         optimizer_fairD_gender_user.step()
    #     print(np.mean(loss[:-1]))
    # torch.save(fairD_gender_user.state_dict(), './preprocessed/GCN_re2_GD.pt')
    # regression_gender_item.train()
    # print('model R')
    # for epoch in range(20):
    #     loss = []
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
    #         regression_gender_item.zero_grad()
    #         l_penalty_user = regression_gender_item(item_embedding.detach(), i, True)
    #         l_penalty_user.backward()
    #         optimizer_regression_gender_item.step()
    #         loss.append(l_penalty_user.item())
    #     print(np.mean(loss[:-1]))
    # torch.save(regression_gender_item.state_dict(), './preprocessed/GCN_re2_GR.pt')
    # pdb.set_trace()


    model.load_state_dict(torch.load("./preprocessed/GCN_re2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/GCN_re2_GD.pt"))
    regression_gender_item.load_state_dict(torch.load("./preprocessed/GCN_re2_GR.pt"))
    fairD_age_user.load_state_dict(torch.load("./preprocessed/GCN_re2_AD.pt"))
    regression_age_item.load_state_dict(torch.load("./preprocessed/GCN_re2_AR.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum4 = []
        train_loss_sum5 = []
        train_loss_sum6 = []
        age_loss_d = []
        age_loss_r = []
        age_loss_d_ = []
        age_loss_r_ = []
        f1_micro_, acc_, f1_ = [], [], []
        gender_AUC_, gender_acc_ = [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            freeze_model(regression_gender_item)
            freeze_model(fairD_age_user)
            freeze_model(regression_age_item)
            fairD_gender_user.eval()
            regression_gender_item.eval()
            fairD_age_user.eval()
            regression_age_item.eval()
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss, loss2, gen_d_loss, gen_r_loss, age_d_loss, age_r_loss = model(u, i, r, GenDisc=fairD_gender_user,
                                                                                     GenReg=regression_gender_item,
                                                                                     AgeDisc=fairD_age_user,
                                                                                     AgeReg=regression_age_item)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(gen_d_loss.item())
            train_loss_sum4.append(gen_r_loss.item())
            age_loss_d.append(age_d_loss.item())
            age_loss_r.append(age_r_loss.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True, GenDisc=fairD_gender_user,
                                                       GenReg=regression_gender_item,
                                                       AgeDisc=fairD_age_user, AgeReg=regression_age_item)

            unfreeze_model(fairD_gender_user)
            unfreeze_model(regression_gender_item)
            unfreeze_model(fairD_age_user)
            unfreeze_model(regression_age_item)
            '''gender'''
            for _ in range(10):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())
            for _ in range(10):
                regression_gender_item.train()
                regression_gender_item.zero_grad()
                l_penalty_item = regression_gender_item(item_embedding.detach(), i, True)
                l_penalty_item.backward()
                optimizer_regression_gender_item.step()
                if _ == 0:
                    train_loss_sum6.append(l_penalty_item.item())
            '''age'''
            for _ in range(40):
                fairD_age_user.train()
                fairD_age_user.zero_grad()
                l_penalty_user = fairD_age_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_age_user.step()
                if _ == 0:
                    age_loss_d_.append(l_penalty_user.item())

            for _ in range(40):
                regression_age_item.train()
                regression_age_item.zero_grad()
                l_penalty_item = regression_age_item(item_embedding.detach(), i, True)
                l_penalty_item.backward()
                optimizer_regression_age_item.step()
                if _ == 0:
                    age_loss_r_.append(l_penalty_item.item())

            with torch.no_grad():
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True, GenDisc=fairD_gender_user,
                                                       GenReg=regression_gender_item,
                                                       AgeDisc=fairD_age_user, AgeReg=regression_age_item)
                dataset_len = batch_size
                y_hat, y = fairD_age_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                y_hat, y = y_hat.data.cpu().numpy(), y.data.cpu().numpy()
                preds = np.argmax(y_hat, axis=-1)
                correct = np.sum(preds == y)
                age_correct.append(correct)
                # pdb.set_trace()
                AUC, acc, f1_micro, f1_macro = metrics(preds, y, y_hat, dataset_len, correct)
                f1_micro_.append(f1_micro)
                acc_.append(acc)

            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                # age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True, GenDisc=fairD_gender_user,
                                                       GenReg=regression_gender_item,
                                                       AgeDisc=fairD_age_user, AgeReg=regression_age_item)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(y.view_as(preds)).sum().item()
                preds_list4.append(preds)
                probs_list4.append(y_hat)
                labels_list4.append(y)
                # age_correct.append(correct)
                GAUC, Gacc, Gf1, Gf1_macro = metrics_gender(preds_list4, labels_list4, probs_list4, dataset_len,
                                                            correct)
                gender_AUC_.append(GAUC)
                gender_acc_.append(Gacc)

        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g6'
        PATH_model = path_save_model_b + '/fair' + str(epoch) + '_com.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        with torch.no_grad():
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss4 = round(np.mean(train_loss_sum4[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        train_loss6 = round(np.mean(train_loss_sum6[:-1]), 4)
        th_d = round(np.mean(age_loss_d_[:-1]), 4)
        main_d = round(np.mean(age_loss_d[:-1]), 4)
        main_r = round(np.mean(age_loss_r[:-1]), 4)
        th_r = round(np.mean(age_loss_r_[:-1]), 4)
        f1_micro_mean = round(np.mean(f1_micro_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)

        gender_acc_mean = round(np.mean(gender_acc_[:-1]), 4)
        gender_AUC_mean = round(np.mean(gender_AUC_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "Gender epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t r loss1:" + str(
            train_loss4) + "\t r loss2:" + str(train_loss6) + "\t acc:" + str(gender_acc_mean) + "\t AUC:" + str(
            gender_AUC_mean)

        str_print_train_age = "Age epoch:" + str(epoch) + ' time:' + str(
            round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            main_d) + '\t user_cls2:' + str(th_d) + "\t r loss1:" + str(
            main_r) + "\t r loss2:" + str(th_r) + "\t acc:" + str(acc_mean) + "\t f1:" + str(f1_micro_mean)
        print(' train_time----', elapsed_time)
        print(str_print_train)
        print(str_print_train_age)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4)) + '\n'  # + '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(
        # round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def baseline_icml_AGE():
    print('------------training processing--------------')
    with open('./preprocessed/ml-1m_gcn_re2.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
        # # age_data = dataset['age_data']
        item_age_inds = dataset['item_age_inds']
        age_labels = dataset['age_labels']
    age_labels = np.array(age_labels)
    item_age_inds = np.array(item_age_inds)
    # pdb.set_trace()
    print('Load complete')

    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    item_inds = np.array(item_inds)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list,item_size)
    sparse_u_i=readTrainSparseMatrix(train_user_list, True, user_d, item_d,user_size)
    sparse_i_u=readTrainSparseMatrix(train_item_list, False,user_d,item_d,item_size)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g7'
    path_save_model = './model/' + dataset_name + '/g7'
    result_file = open(path_save_log_base + '/ICML_AGE.txt', 'a')

    model = GCN_icml_age(user_size, item_size, factor_num,sparse_u_i,sparse_i_u,user_d,item_d,avg_rating).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = AgeDiscriminator(factor_num, age_labels).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    ################
    #   pretrain   #
    ################
    model.train()
    fairD_gender_user.train()
    model.load_state_dict(torch.load("./preprocessed/GCN_re2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/GCN_re2_AD.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum5 = []
        AUC_, acc_, f1_ = [], [], []
        f1_micro_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            fairD_gender_user.eval()

            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss, loss2, d_loss, r_loss = model(u, i, r, Disc=fairD_gender_user)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)

            unfreeze_model(fairD_gender_user)
            for _ in range(40):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())


            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       Disc=fairD_gender_user)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                y_hat, y = y_hat.data.cpu().numpy(), y.data.cpu().numpy()
                preds = np.argmax(y_hat, axis=-1)
                correct = np.sum(preds == y)
                age_correct.append(correct)
                AUC, acc, f1_micro, f1_macro = metrics(preds, y, y_hat, dataset_len, correct)
                f1_micro_.append(f1_micro)
                acc_.append(acc)

        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g7'
        PATH_model = path_save_model_b + '/icml_age_' + str(epoch) + '.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        # training finish, we want to cum fairness and rmse
        with torch.no_grad():
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        f1_micro_mean = round(np.mean(f1_micro_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t acc:" + str(acc_mean) + "\t f1:" + str(f1_micro_mean)

        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def baseline_icml_com():
    print('------------training processing--------------')
    with open('./preprocessed/ml-1m_gcn_re2.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
        # # age_data = dataset['age_data']
        item_age_inds = dataset['item_age_inds']
        age_labels = dataset['age_labels']
    age_labels = np.array(age_labels)
    item_age_inds = np.array(item_age_inds)
    # pdb.set_trace()
    print('Load complete')

    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    item_inds = np.array(item_inds)
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list,item_size)
    sparse_u_i=readTrainSparseMatrix(train_user_list, True, user_d, item_d,user_size)
    sparse_i_u=readTrainSparseMatrix(train_item_list, False,user_d,item_d,item_size)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=val_set_len, shuffle=False)
    path_save_log_base = './Logs/' + dataset_name + '/g8'
    path_save_model = './model/' + dataset_name + '/g8'
    result_file = open(path_save_log_base + '/icml_com.txt', 'a')

    model = GCN_icml_com(user_size, item_size, factor_num,sparse_u_i,sparse_i_u,user_d,item_d,avg_rating).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    fairD_age_user = AgeDiscriminator(factor_num, age_labels).cuda()
    optimizer_fairD_age_user = torch.optim.Adam(fairD_age_user.parameters(), lr=0.005)
    ################
    #   pretrain   #
    ################
    model.train()
    fairD_gender_user.train()
    fairD_age_user.train()
    model.load_state_dict(torch.load("./preprocessed/GCN_re2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/GCN_re2_GD.pt"))
    fairD_age_user.load_state_dict(torch.load("./preprocessed/GCN_re2_AD.pt"))
    ################
    #  train part  #
    ################
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum5 = []
        d_loss_age_sum = []
        d_loss_age_sum2 = []
        f1_micro_, acc_, f1_ = [], [], []
        gender_AUC_, gender_acc_ = [], []
        for idx, (u, i, r) in enumerate(train_loader):
            model.train()
            freeze_model(fairD_gender_user)
            freeze_model(fairD_age_user)
            fairD_gender_user.eval()
            fairD_age_user.eval()
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            model.zero_grad()
            task_loss,loss2,d_loss,d_loss_age  = model(u, i, r,GenDisc=fairD_gender_user,AgeDisc=fairD_age_user)
            task_loss.backward()
            optimizer.step()

            train_loss_sum.append(loss2.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())
            d_loss_age_sum.append(d_loss_age.item())

            with torch.no_grad():
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       GenDisc=fairD_gender_user,
                                                       AgeDisc=fairD_age_user)

            unfreeze_model(fairD_gender_user)
            unfreeze_model(fairD_age_user)
            fairD_gender_user.train()
            fairD_age_user.train()
            for _ in range(10):
                fairD_gender_user.train()
                fairD_gender_user.zero_grad()

                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())

            for _ in range(40):
                fairD_age_user.zero_grad()
                l_penalty_user = fairD_age_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_age_user.step()
                if _ == 0:
                    d_loss_age_sum2.append(l_penalty_user.item())

            with torch.no_grad():
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       GenDisc=fairD_gender_user,
                                                       AgeDisc=fairD_age_user)
                dataset_len = batch_size
                y_hat, y = fairD_age_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                y_hat, y = y_hat.data.cpu().numpy(), y.data.cpu().numpy()
                preds = np.argmax(y_hat, axis=-1)
                correct = np.sum(preds == y)
                age_correct.append(correct)
                # pdb.set_trace()
                AUC, acc, f1_micro, f1_macro = metrics(preds, y, y_hat, dataset_len, correct)
                f1_micro_.append(f1_micro)
                acc_.append(acc)

            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True,
                                                       GenDisc=fairD_gender_user,
                                                       AgeDisc=fairD_age_user)
                dataset_len = batch_size
                y_hat, y = fairD_gender_user.predict(user_embedding.detach(), u, return_loss=False, return_preds=False)
                preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(y.view_as(preds)).sum().item()
                preds_list4.append(preds)
                probs_list4.append(y_hat)
                labels_list4.append(y)
                age_correct.append(correct)
                GAUC, Gacc, Gf1, Gf1_macro = metrics_gender(preds_list4, labels_list4, probs_list4, dataset_len,
                                                            correct)
                gender_AUC_.append(GAUC)
                gender_acc_.append(Gacc)


        # pdb.set_trace()
        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/g8'
        PATH_model = path_save_model_b + '/icml_com_' + str(epoch) + '.pt'  # -15
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        fairD_gender_user.eval()
        fairD_age_user.eval()
        # training finish, we want to cum fairness and rmse
        with torch.no_grad():
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse = torch.sqrt(rmse)
            val_rmse = val_rmse.item()
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse = torch.sqrt(rmse)
            test_rmse = test_rmse.item()
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)

        f1_micro_mean = round(np.mean(f1_micro_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)

        gender_acc_mean = round(np.mean(gender_acc_[:-1]), 4)
        gender_AUC_mean = round(np.mean(gender_AUC_[:-1]), 4)

        age_loss1 = round(np.mean(d_loss_age_sum[:-1]), 4)
        age_loss2 = round(np.mean(d_loss_age_sum2[:-1]), 4)

        elapsed_time = time.time() - start_time

        str_print_train = "Gender epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            train_loss3) + '\t user_cls2:' + str(train_loss5) + "\t acc:" + str(gender_acc_mean) + "\t AUC:" + str(
            gender_AUC_mean)
        str_print_train_age = "Age epoch:" + str(epoch) + ' time:' + str(
            round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" + str(
            age_loss1) + '\t user_cls2:' + str(age_loss2) + "\t acc:" + str(acc_mean) + "\t AUC:" + str(f1_micro_mean)
        print(' train_time----', elapsed_time)
        print(str_print_train)
        print(str_print_train_age)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')
        result_file.write('\n')
        result_file.flush()

def test_new_cls_and_reg2():
    # print('------------test embeds_cls_accuracy processing--------------')
    # # Load preprocess data
    # # with open('./preprocessed/ml-1m_gcn.pickle', 'rb') as f:
    # with open('./preprocessed/ml-1m_gcn_rebuttal.pickle', 'rb') as f:
    #     dataset = pickle.load(f)
    #     user_size, item_size = dataset['user_size'], dataset['item_size']
    #     train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
    #         'test_user_list']
    #     all_user_list = dataset['all_user_list']
    #     train_pair = dataset['train_pair']
    #     test_pair = dataset['test_pair']
    #     val_pair = dataset['val_pair']
    #     gender_data = dataset['gender_data']
    # print('Load complete')
    print("fair com")
    with open('./preprocessed/ml-1m_gcn_re2.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
        # # age_data = dataset['age_data']
        item_age_inds = dataset['item_age_inds']
        age_labels = dataset['age_labels']
    age_labels = np.array(age_labels)
    item_age_inds = np.array(item_age_inds)
    # pdb.set_trace()
    print('Load complete')

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            train_item_list[item][user] = rating
    '''
    read D
    '''
    user_d = readD(train_user_list, user_size)
    item_d = readD(train_item_list, item_size)
    sparse_u_i = readTrainSparseMatrix(train_user_list, True, user_d, item_d, user_size)
    sparse_i_u = readTrainSparseMatrix(train_item_list, False, user_d, item_d, item_size)

    # model = GCN_bias_faircf(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-100k/g5/fair93_re2.pt'

    # model = GCN_compostional(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d,
    #                          avg_rating).cuda()
    # Path_model = './model/ml-100k/g6/fair60_com.pt'
    # model = GCN_bias(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/GCN_bias_36(2).pt'

    # model = GCN_bias_faircf(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/2adv_42.pt'
    # Path_model = './model/ml-100k/g0/rebuttal_38.pt'

    # model = GCN_bias_faircf_user(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d,
    #                              avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/cfu_124.pt'
    #
    # model = GCN_icml_2019(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-1m/g0/icml_34.pt'

    # model = GCN_icml_age(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    # Path_model = './model/ml-100k/g7/icml_age_30.pt'

    model = GCN_icml_com(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, user_d, item_d, avg_rating).cuda()
    Path_model = './model/ml-100k/g8/icml_com_57.pt'
    model.train()
    model.load_state_dict(torch.load(Path_model))
    model.eval()
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    fairD_age_user = AgeDiscriminator(factor_num, age_labels).cuda()
    optimizer_fairD_age_user = torch.optim.Adam(fairD_age_user.parameters(), lr=0.005)

    # regression_gender_item = GenderRegression_onedim(factor_num, item_rating).cuda()
    # optimizer_regression_gender_item = torch.optim.Adam(regression_gender_item.parameters(), lr=0.01)

    users = np.arange(user_size)
    np.random.seed(0)
    np.random.shuffle(users)
    cut = int(user_size * 0.8)
    train_users = users[:cut]
    test_users = users[cut:]
    train_users_index = np.array(train_users)
    test_users_index = np.array(test_users)
    train_len = len(train_users_index)
    test_len = len(test_users_index)

    model.train()
    users_embedding, items_embedding, user_bias, item_bias = model.predict(torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]),
                                                     torch.cuda.LongTensor([0]), return_e=True)
    all_user_embedding = users_embedding.detach()
    all_item_embedding = items_embedding.detach()
    model.eval()
    # pdb.set_trace()
    '''
    Gender
    '''
    x___ = 1
    if x___ == 1:
        for epoch in tqdm(range(200)):
            fairD_gender_user.train()
            p_batch = torch.cuda.LongTensor(train_users_index)
            filter_l_emb = all_user_embedding[p_batch]
            l_penalty_2 = fairD_gender_user(filter_l_emb.detach(), p_batch, return_loss=True)
            optimizer_fairD_gender_user.zero_grad()
            l_penalty_2.backward()
            optimizer_fairD_gender_user.step()
            train_loss_sum = l_penalty_2.item()
            # pdb.set_trace()

            fairD_gender_user.eval()
            with torch.no_grad():
                dataset_len = train_len
                preds_list4, labels_list4, probs_list4 = [], [], []
                y_hat, y = fairD_gender_user.predict(filter_l_emb.detach(), p_batch, return_loss=False, return_preds=False)
                preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(y.view_as(preds)).sum().item()
                preds_list4.append(preds)
                probs_list4.append(y_hat)
                labels_list4.append(y)
                AUC, acc, f1, f1_macro = metrics_gender(preds_list4, labels_list4, probs_list4, dataset_len, correct)

                # testing set's correct
                p_batch_2 = torch.cuda.LongTensor(test_users_index)
                filter_l_emb_2 = all_user_embedding[p_batch_2]
                dataset_len = test_len
                preds_list, labels_list, probs_list = [], [], []
                y_hat_2, l_A_labels = fairD_gender_user.predict(filter_l_emb_2.detach(), p_batch_2, return_loss=False,
                                                                return_preds=False)
                test_loss = fairD_gender_user.predict(filter_l_emb_2.detach(), p_batch_2, return_loss=True)
                test_loss = test_loss.item()

                preds = (y_hat_2 > torch.Tensor([0.5]).cuda()).float() * 1
                l_correct = preds.eq(l_A_labels.view_as(preds)).sum().item()
                preds_list.append(preds)
                probs_list.append(y_hat_2)
                labels_list.append(l_A_labels)
                AUC1, acc1, f1, f1_macro1 = metrics_gender(preds_list, labels_list, probs_list, dataset_len, l_correct)

                str_print_test = "Train auc is: %f Test auc is: %f loss is: %f test loss: %f test acc: %f" % (
                AUC, AUC1, train_loss_sum, test_loss, acc1)
                print(str_print_test)
    else:
        '''
        AGE
        '''
        for epoch in tqdm(range(200)):
            fairD_age_user.train()
            p_batch = torch.cuda.LongTensor(train_users_index)
            filter_l_emb= all_user_embedding[p_batch]
            l_penalty_2 = fairD_age_user(filter_l_emb.detach(), p_batch, return_loss=True)
            optimizer_fairD_age_user.zero_grad()
            l_penalty_2.backward()
            optimizer_fairD_age_user.step()
            train_loss_sum = l_penalty_2.item()
            # pdb.set_trace()

            fairD_age_user.eval()
            with torch.no_grad():
                dataset_len = train_len
                y_hat, y = fairD_age_user.predict(filter_l_emb.detach(), p_batch, return_loss=False, return_preds=False)
                y_hat, y = y_hat.data.cpu().numpy(), y.data.cpu().numpy()
                preds = np.argmax(y_hat, axis=-1)
                # pdb.set_trace()
                correct = np.sum(preds == y)
                AUC, acc, f1_micro, f1_macro = metrics(preds, y, y_hat, dataset_len, correct)


                # testing set's correct
                p_batch_2 = torch.cuda.LongTensor(test_users_index)
                filter_l_emb_2 = all_user_embedding[p_batch_2]
                dataset_len = test_len
                y_hat, y = fairD_age_user.predict(filter_l_emb_2.detach(), p_batch_2, return_loss=False, return_preds=False)
                y_hat, y = y_hat.data.cpu().numpy(), y.data.cpu().numpy()
                preds = np.argmax(y_hat, axis=-1)
                # pdb.set_trace()
                correct = np.sum(preds == y)
                AUC1, acc1, f1_micro1, f1_macro1 = metrics(preds, y, y_hat, dataset_len, correct)


                str_print_test = "Train f1 is: %f Test f1 is: %f loss is: %f test acc: %f" % (f1_micro,f1_micro1,train_loss_sum,acc1)
                print(str_print_test)
            # with torch.no_grad():
            #     dataset_len = train_len
            #     preds_list4, labels_list4, probs_list4 = [], [], []
            #     y_hat, y = fairD_gender_user.predict(filter_l_emb.detach(), p_batch,return_loss=False, return_preds=False)
            #     preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            #     correct = preds.eq(y.view_as(preds)).sum().item()
            #     preds_list4.append(preds)
            #     probs_list4.append(y_hat)
            #     labels_list4.append(y)
            #     AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)
            #
            #     # testing set's correct
            #     p_batch_2 = torch.cuda.LongTensor(test_users_index)
            #     filter_l_emb_2 = all_user_embedding[p_batch_2]
            #     dataset_len = test_len
            #     preds_list, labels_list, probs_list = [], [], []
            #     y_hat_2, l_A_labels = fairD_gender_user.predict(filter_l_emb_2.detach(), p_batch_2,return_loss=False, return_preds=False)
            #     test_loss = fairD_gender_user.predict(filter_l_emb_2.detach(), p_batch_2, return_loss=True)
            #     test_loss = test_loss.item()
            #
            #     preds = (y_hat_2 > torch.Tensor([0.5]).cuda()).float() * 1
            #     l_correct = preds.eq(l_A_labels.view_as(preds)).sum().item()
            #     preds_list.append(preds)
            #     probs_list.append(y_hat_2)
            #     labels_list.append(l_A_labels)
            #     AUC1, acc1, f1, f1_macro1 = metrics(preds_list, labels_list, probs_list, dataset_len, l_correct)
            #
            #     str_print_test = "Train auc is: %f Test auc is: %f loss is: %f test loss: %f test acc: %f" % (AUC,AUC1,train_loss_sum,test_loss,acc1)
            #     print(str_print_test)

if __name__ == '__main__':
    print(run_id)
    # baseline_faircf()
    # baseline_faircf_user()
    # baseline_icml_2019()

    # baseline_fair_age()
    # baseline_fair_com()
    #
    test_new_cls_and_reg2()
    # test_MF()
    # baseline_icml_AGE()
    # baseline_icml_com()
