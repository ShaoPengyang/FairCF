#-- coding:UTF-8 --
'''
train RMSE with lastfm-360k 2020/5/13
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
from torch.nn.utils import clip_grad_norm
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
from tqdm import tqdm
tqdm.monitor_interval = 0
from models import *
from evaluate import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="change gpuid")
parser.add_argument("-g", "--gpu-id", help="choose which gpu to use", type=str, default=str(3))
parser.add_argument("-t", "--times", help="choose times", type=str, default='0')
parser.add_argument("-d", "--D_steps", help="num of train Disc times", type=int, default=2)
parser.add_argument("-b", "--batch", help="num of train Disc times", type=int, default=65536)
parser.add_argument("--lr", help="num of train Disc times", type=float, default=0.005)
# 0 epoch 41 no adversary
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
D_steps = args.D_steps
run_id = 't' + args.times
dataset_name = 'lastfm'
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
user_num = 359347
item_num = 292589
factor_num = 64
batch_size = 1024*1024
# batch_size = 1024*1024
print(batch_size)
learning_rate = args.lr

# def fair_K_lastfm_bias__(user_e,item_e,user_bias,item_bias,gender_data,train_user_list):
#     '''
#     compute rethinking_fairness@ALL among random picked 10000 items
#     step1: for every user, get their rating and divide due to gender
#     step2: for every item, cal rating
#     '''
#     begin_eval=time.time()
#     gender_data = np.array(gender_data)
#     # pdb.set_trace()
#     all_prediction = np.matmul(user_e, item_e.T) + user_bias + item_bias
#     # pdb.set_trace()
#     # male_rating = all_prediction*(gender_data.reshape(-1,1))
#     # female_rating = all_prediction*((1-gender_data).reshape(-1,1))
#     #
#     # male_rating = male_rating.sum(0)/(gender_data.sum())
#     # female_rating = female_rating.sum(0)/((1-gender_data).sum())
#     # fairness_rethink_K_all = (np.fabs(male_rating-female_rating)).mean()
#     # pdb.set_trace()
#     topk = 50
#     set_list_all = set(range(2500))
#     item_female = defaultdict(list)
#     item_male = defaultdict(list)
#     item_female_all = defaultdict(list)
#     item_male_all = defaultdict(list)
#     for userid in range(user_e.shape[0]):
#         item_i_list = list(set_list_all - set(train_user_list[userid].keys()))
#         if len(item_i_list) > 2500:
#             pdb.set_trace()
#         # assert len(item_i_list) > 2500, "more than 2500"
#         for item_id in item_i_list:
#             if gender_data[userid] == 0:
#                 item_male_all[item_id].append(all_prediction[userid][item_id])
#             else:
#                 item_female_all[item_id].append(all_prediction[userid][item_id])
#         prediction = all_prediction[userid][item_i_list]
#         indices1 = largest_indices(prediction, topk)
#         # pdb.set_trace()
#         indices1 = list(indices1[0])
#         item_i_list = np.array(item_i_list)
#         top_k_index = item_i_list[indices1]
#         # pdb.set_trace()
#         for i in top_k_index:
#             if gender_data[userid] == 0:
#                 item_male[i].append(all_prediction[userid][i])
#             else:
#                 item_female[i].append(all_prediction[userid][i])
#     fairness_rethink_K2 = 0
#     fairness_rethink_K_all = 0
#     length = item_e.shape[0]
#     for i in range(item_e.shape[0]):
#         if len(item_male[i]) == 0 and len(item_female[i]) == 0:
#             extra = 0
#         elif len(item_male[i]) == 0:
#             extra = np.mean(item_female[i])
#         elif len(item_female[i]) == 0:
#             extra = np.mean(item_male[i])
#         else:
#             male_mean_rating_i = np.mean(item_male[i])
#             female_mean_rating_i = np.mean(item_female[i])
#             extra = male_mean_rating_i - female_mean_rating_i
#             # pdb.set_trace()
#         fairness_rethink_K2 += np.fabs(extra)
#     fairness_rethink_K2 = fairness_rethink_K2 / length
#
#     for i in range(item_e.shape[0]):
#         if len(item_male_all[i]) == 0 and len(item_female_all[i]) == 0:
#             extra = 0
#         elif len(item_male_all[i]) == 0:
#             extra = np.mean(item_female_all[i])
#         elif len(item_female_all[i]) == 0:
#             extra = np.mean(item_male_all[i])
#         else:
#             male_mean_rating_i = np.mean(item_male_all[i])
#             female_mean_rating_i = np.mean(item_female_all[i])
#             extra = male_mean_rating_i - female_mean_rating_i
#             # pdb.set_trace()
#         fairness_rethink_K_all += np.fabs(extra)
#     fairness_rethink_K_all = fairness_rethink_K_all / length
#     print(length)
#     print("fairness@all end time:\t" + str(round((time.time() - begin_eval), 4)))
#     return fairness_rethink_K_all,fairness_rethink_K2

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
    # pdb.set_trace()
    # male_rating = all_prediction*(gender_data.reshape(-1,1))
    # female_rating = all_prediction*((1-gender_data).reshape(-1,1))
    #
    # male_rating = male_rating.sum(0)/(gender_data.sum())
    # female_rating = female_rating.sum(0)/((1-gender_data).sum())
    # fairness_rethink_K_all = (np.fabs(male_rating-female_rating)).mean()
    # pdb.set_trace()
    set_list_all = set(range(2500))
    item_female = defaultdict(list)
    item_male = defaultdict(list)
    # item_female_all = defaultdict(list)
    # item_male_all = defaultdict(list)
    for userid in range(user_e.shape[0]):
        item_i_list = list(set_list_all - set(train_user_list[userid].keys()))
        if len(item_i_list) > 2500:
            pdb.set_trace()
        # assert len(item_i_list) > 2500, "more than 2500"
        # for item_id in item_i_list:
        #     if gender_data[userid] == 0:
        #         item_male_all[item_id].append(all_prediction[userid][item_id])
        #     else:
        #         item_female_all[item_id].append(all_prediction[userid][item_id])
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

    # extra = 0
    # for i in range(item_e.shape[0]):
    #     if len(item_male_all[i]) == 0 and len(item_female_all[i]) == 0:
    #         extra = 0
    #     elif len(item_male_all[i]) == 0:
    #         extra = np.mean(item_female_all[i])
    #     elif len(item_female_all[i]) == 0:
    #         extra = np.mean(item_male_all[i])
    #     else:
    #         male_mean_rating_i = np.mean(item_male_all[i])
    #         female_mean_rating_i = np.mean(item_female_all[i])
    #         extra = male_mean_rating_i - female_mean_rating_i
    #         # pdb.set_trace()
    #     fairness_rethink_K_all += np.fabs(extra)
    # fairness_rethink_K_all = fairness_rethink_K_all / length
    print(length)
    print("fairness@all end time:\t" + str(round((time.time() - begin_eval), 4)))
    return fairness_rethink_K_all,fairness_rethink_K2

def fair_K_lastfm_bias(user_e,item_e,user_bias,item_bias,gender_data,train_user_list):
    '''
    compute rethinking_fairness@ALL among random picked 10000 items
    step1: for every user, get their rating and divide due to gender
    step2: for every item, cal rating
    '''
    begin_eval = time.time()
    gender_data = np.array(gender_data)
    # pdb.set_trace()
    all_prediction = np.matmul(user_e, item_e.T) + user_bias + item_bias

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

def new_fairness_all(user_e,item_e,user_bias,item_bias,gender_data):
    start_time = time.time()
    gender_data = np.array(gender_data)
    num_gender1 = np.sum(gender_data)
    num_gender2 = user_num - num_gender1
    gender_index1 = np.where(gender_data == 1)
    gender_index2 = np.where(gender_data == 0)
    value = 0
    # pdb.set_trace()
    for id in range(item_num):
        tmp_user_e = user_e
        tmp_item_e = item_e[id].reshape(-1, 1)
        all_prediction = np.matmul(tmp_user_e, tmp_item_e) + user_bias
        prediction1 = all_prediction[gender_index1]
        prediction2 = all_prediction[gender_index2]
        tmp_value = np.fabs(np.sum(prediction1)/num_gender1 - np.sum(prediction2)/num_gender2)
        value += tmp_value
        print(time.time() - start_time)
    value = value/item_num
    spent_time = time.time() - start_time
    print(spent_time)
    return value

def baseline_faircf():
    print('------------training processing--------------')
    time1 = time.time()
    with open('./preprocessed/last_fm_preprocessed.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        item_inds = dataset['item_inds']
    # pdb.set_trace()
    time2 = time.time()
    print('Load complete, use time: '+str(time2-time1))

    item_inds = np.array(item_inds)
    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:,2])
    print("avg:"+str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=65536, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=65536, shuffle=False)

    model = BPR_2_filter_bias(user_size, item_size, factor_num, avg_rating).cuda()
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    regression_gender_item = GenderRegression_onedim(factor_num, item_inds).cuda()

    optimizer_regression_gender_item = torch.optim.Adam(regression_gender_item.parameters(), lr=0.005)
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


    log_path = path_save_log_base + '/date_1202_bias_2adv_PMF11' + '.txt'
    print(log_path)
    result_file = open(log_path, 'a')
    result_file.write('\n\n')
    result_file.write('train MF on lastfm')


    model.train()
    fairD_gender_user.train()
    regression_gender_item.train()
    # print("pre train base model")
    # for epoch in range(10):
    #     loss_sum = []
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         model.train()
    #         fairD_gender_user.eval()
    #         regression_gender_item.eval()
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         optimizer.zero_grad()
    #         task_loss, loss2, d_loss, l2_loss = model(u, i, r, Discrminator=None)
    #         task_loss.backward()
    #         optimizer.step()
    #         loss_sum.append(task_loss.item())
    #     loss = round(np.mean(loss_sum[:-1]), 4)
    #     print(loss)
    # torch.save(model.state_dict(), './preprocessed/recommend_pretrain_2.pt')
    # print("pre train base disc")
    # for epoch in range(20):
    #     d_loss_sum = []
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         model.eval()
    #         fairD_gender_user.train()
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True, Discrminator=None)
    #         fairD_gender_user.zero_grad()
    #         l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
    #         l_penalty_user.backward()
    #         optimizer_fairD_gender_user.step()
    #         d_loss_sum.append(l_penalty_user.item())
    #     d_loss = round(np.mean(d_loss_sum[:-1]), 4)
    #     print(d_loss)
    # torch.save(fairD_gender_user.state_dict(), './preprocessed/DISC_pretrain_2.pt')
    # print("pre train base reg")
    # for epoch in range(20):
    #     r_loss_sum = []
    #     for idx, (u, i, r) in enumerate(train_loader):
    #         regression_gender_item.train()
    #         u = u.cuda()
    #         i = i.cuda()
    #         r = r.cuda()
    #         with torch.no_grad():
    #             user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True, Discrminator=None)
    #         regression_gender_item.zero_grad()
    #         l_penalty_item = regression_gender_item(item_embedding.detach(), i, True)
    #         l_penalty_item.backward()
    #         optimizer_regression_gender_item.step()
    #         r_loss_sum.append(l_penalty_item.item())
    #     r_loss = round(np.mean(r_loss_sum[:-1]), 4)
    #     print(r_loss)
    # torch.save(regression_gender_item.state_dict(), './preprocessed/REG_pretrain_2.pt')
    #
    # pdb.set_trace()
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_2.pt"))
    regression_gender_item.load_state_dict(torch.load("./preprocessed/REG_pretrain_2.pt"))

    for epoch in range(200):
        model.train()
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum4 = []
        train_loss_sum5 = []
        train_loss_sum6 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            fairD_gender_user.eval()
            regression_gender_item.eval()
            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            optimizer.zero_grad()
            task_loss, recommend_loss, d_loss, r_loss = model(u, i, r, Discrminator=fairD_gender_user,Regression=regression_gender_item)
            task_loss.backward()
            optimizer.step()
            train_loss_sum.append(recommend_loss.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())
            train_loss_sum4.append(r_loss.item())

            D_steps = 20
            for _ in range(D_steps):
                fairD_gender_user.train()
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
                optimizer_fairD_gender_user.zero_grad()
                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum6.append(l_penalty_user.item())
            for _ in range(20):
                regression_gender_item.train()
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
                optimizer_regression_gender_item.zero_grad()
                l_penalty_item = regression_gender_item(item_embedding.detach(), i, True)
                l_penalty_item.backward()
                optimizer_regression_gender_item.step()
                if _ == 0:
                    train_loss_sum5.append(l_penalty_item.item())

            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
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

        ####################
        #    save model    #
        ####################
        # 21 - 149
        PATH_model=path_save_model_base+'/2adv_5bias'+str(epoch)+'.pt'  # -15 -100
        str_save = 'save this epoch\'s model to this path:' + PATH_model + '\n'
        result_file.write(str_save)
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        ###############
        #  eval part  #
        ###############
        fairness, nips2017 = 0, 0
        with torch.no_grad():
            val_rmse = 0
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse += rmse.item()
                # pdb.set_trace()
            val_rmse = np.sqrt(val_rmse / val_set_len)
            test_rmse = 0
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse += rmse.item()
            test_rmse = np.sqrt(test_rmse / test_set_len)

            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            user_e = user_e[:10000]
            item_e = item_e[:2500]
            user_bias_tmp = user_bias_tmp[:10000]
            item_bias_tmp = item_bias_tmp[:, :2500]
            gender_data = gender_data[:10000]
            fairness_all, fairness50 = fair_K_lastfm_bias(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                          train_user_list)

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
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" +str(
            train_loss3)+ '\t cls_loss2:'+str(train_loss6) + "\t r_loss:" + str(train_loss4) + '\t r loss2:' + str(train_loss5) + "\t acc:" + str(
            acc_mean) + "\t AUC:" + str(AUC_mean)
        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_test = '\t val rmse' + str(val_rmse) + '\t test rmse:' + str(test_rmse) + '\t fair 50:'+str(fairness50)+'\t fair all:'+str(fairness_all)
        print(str_print_test)
        result_file.write(str_print_test)
        result_file.write('\n')
        result_file.flush()

def test_MF():
    topk = 50
    # Load preprocess data
    with open('./preprocessed/last_fm_preprocessed.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['all_user_list'], dataset[
            'test_user_list']
        all_user_list = dataset['all_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)

    train_set_len, test_set_len, val_set_len = len(train_pair), len(test_pair), len(val_pair)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=65536, shuffle=False)
    # model = BPR_2_filter_bias(user_size, item_size, factor_num, avg_rating).cuda()
    # Path_model = './model/lastfm/t0/2adv_bias_6158.pt'

    model = BPR_icml_bias(user_size, item_size, factor_num, avg_rating).cuda()
    Path_model = './model/lastfm/t0/icml_188.pt'

    # model = BPR_bias_faircf_user(user_size, item_size, factor_num, avg_rating).cuda()
    # Path_model = './model/lastfm/t0/cfu_173.pt'
    print(Path_model)
    model.train()
    model.load_state_dict(torch.load(Path_model))
    model.eval()
    with torch.no_grad():
        test_rmse = 0
        for idx, (u, i, r) in enumerate(test_loader):
            u, i, r = u.cuda(), i.cuda(), r.cuda()
            rmse = model.predict(u, i, r)
            test_rmse += rmse.item()
        test_rmse = np.sqrt(test_rmse / test_set_len)
    users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
    user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
    item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
    user_bias = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
    item_bias = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
    # fairness_All = new_fairness_all(user_e, item_e, user_bias, item_bias, gender_data)
    # print("\t fair@50:" + str(fairness_All))
    user_e = user_e[:10000]
    item_e = item_e[:2500]
    user_bias_tmp = user_bias[:10000]
    item_bias_tmp = item_bias[:, :2500]
    gender_data = gender_data[:10000]
    fairness_all, fairness10 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 10)
    fairness_all, fairness20 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 20)
    fairness_all, fairness30 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 30)
    fairness_all, fairness40 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                    train_user_list, 40)
    rmse_str = "\t test rmse:" + str(test_rmse)
    nips_str = "\t fair@10:" + str(fairness10) + "\t fair@20:" + str(fairness20) + "\t fair@30:" + str(
        fairness30) + "\t fair@40:" + str(fairness40)
    print(rmse_str)
    print(nips_str)
    # fairness_all, fairness50 = fair_K_lastfm_bias__(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
    #                                               train_user_list)
    # rmse_str = "\t test rmse:"+str(test_rmse)
    # nips_str ="\t fair@50:" + str(fairness50) + '\t fair@all:'+str(fairness_all)
    # print(rmse_str)
    # print(nips_str)

def train_newDisc():
    # Load preprocess data
    # with open('../fairness2020426/preprocessed/ml-1m_gcn.pickle', 'rb') as f:
    with open('./preprocessed/last_fm_preprocessed.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
    # pdb.set_trace()

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:, 2])
    print("avg:" + str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)
    print(user_size)
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

    train_dataset = UsersDataset(train_users_index,train_len)
    batchsize = 8192*8
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)

    tmp_list = np.load('./preprocessed/tmp_list.npy')
    model = BPR_2_filter_bias(user_size, item_size, factor_num,avg_rating).cuda()
    # model = BPR_icml_bias(user_size, item_size, factor_num,avg_rating).cuda()
    # PATH_model = './model/lastfm/t0/20200926_bias_6_batchsize_1048576.pt'
    # Path_model = './model/lastfm/t0/icml_189.pt'
    # Path_model = './model/lastfm/t0/cfu_173.pt'
    Path_model = './model/lastfm/t0/2adv_bias_6158.pt'
    print(Path_model)
    # Path_model = './model/lastfm/t0/2adv_5bias110.pt'
    # Path_model = './preprocessed/recommend_pretrain.pt'
    model.train()
    model.load_state_dict(torch.load(Path_model))
    users_embedding, items_embedding,user_bias_tmp,item_bias_tmp = model.predict(0, 0, 0, return_e=True)
    # pdb.set_trace()
    all_user_embedding = users_embedding.detach()
    model.eval()
    for epoch in tqdm(range(200)):
        train_loss_sum=[]
        fairD_gender_user.train()
        for idx, p_batch in enumerate(train_loader):
            p_batch = p_batch.cuda()
            filter_l_emb= all_user_embedding[p_batch]
            l_penalty_2 = fairD_gender_user(filter_l_emb.detach(), p_batch, return_loss=True)
            optimizer_fairD_gender_user.zero_grad()
            l_penalty_2.backward()
            optimizer_fairD_gender_user.step()
            train_loss_sum.append(l_penalty_2.item())
        train_loss_sum = round(np.mean(train_loss_sum[:-1]),4)

        fairD_gender_user.eval()
        with torch.no_grad():
            p_batch = torch.cuda.LongTensor(train_users_index)
            filter_l_emb = all_user_embedding[p_batch]
            dataset_len = train_len
            preds_list4, labels_list4, probs_list4 = [], [], []
            y_hat, y = fairD_gender_user.predict(filter_l_emb.detach(), p_batch,return_loss=False, return_preds=False)
            preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            correct = preds.eq(y.view_as(preds)).sum().item()
            preds_list4.append(preds)
            probs_list4.append(y_hat)
            labels_list4.append(y)
            AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)

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
            # acc1 = l_correct/test_len
            AUC1, acc1, f1, f1_macro1 = metrics(preds_list, labels_list, probs_list, dataset_len, l_correct)

            str_print_test = "Train auc is: %f Test auc is: %f loss is: %f test loss: %f train acc: %f test acc: %f" % (AUC,AUC1,train_loss_sum,test_loss,acc,acc1)
            print(str_print_test)

def baseline_icml():
    print('------------training processing--------------')
    time1 = time.time()
    with open('./preprocessed/last_fm_preprocessed.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
    # pdb.set_trace()
    time2 = time.time()
    print('Load complete, use time: '+str(time2-time1))

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:,2])
    print("avg:"+str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=65536, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=65536, shuffle=False)

    model = BPR_icml_bias(user_size, item_size, factor_num, avg_rating).cuda()
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    log_path = path_save_log_base + '/date_1204_icml_pmf' + '.txt'
    print(log_path)
    result_file = open(log_path, 'a')
    result_file.write('\n\n')
    result_file.write('train MF on lastfm')

    model.train()
    fairD_gender_user.train()
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_2.pt"))

    for epoch in range(200):
        model.train()
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum6 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            fairD_gender_user.eval()

            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            optimizer.zero_grad()
            task_loss, recommend_loss, d_loss, r_loss = model(u, i, r, Discrminator=fairD_gender_user)
            task_loss.backward()
            optimizer.step()
            train_loss_sum.append(recommend_loss.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())


            D_steps = 20
            for _ in range(D_steps):
                fairD_gender_user.train()
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
                optimizer_fairD_gender_user.zero_grad()
                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum6.append(l_penalty_user.item())

            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
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

        ####################
        #    save model    #
        ####################
        # 21 - 149
        PATH_model=path_save_model_base+'/icml_'+str(epoch)+'.pt'  # -15 -100
        str_save = 'save this epoch\'s model to this path:' + PATH_model + '\n'
        result_file.write(str_save)
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        ###############
        #  eval part  #
        ###############
        fairness, nips2017 = 0, 0
        with torch.no_grad():
            val_rmse = 0
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse += rmse.item()
                # pdb.set_trace()
            val_rmse = np.sqrt(val_rmse / val_set_len)
            test_rmse = 0
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse += rmse.item()
            test_rmse = np.sqrt(test_rmse / test_set_len)

            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            user_e = user_e[:10000]
            item_e = item_e[:2500]
            user_bias_tmp = user_bias_tmp[:10000]
            item_bias_tmp = item_bias_tmp[:, :2500]
            gender_data = gender_data[:10000]
            fairness_all, fairness50 = fair_K_lastfm_bias(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                          train_user_list)

        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss6 = round(np.mean(train_loss_sum6[:-1]), 4)
        AUC_mean = round(np.mean(AUC_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" +str(
            train_loss3)+ '\t cls_loss2:'+str(train_loss6)  + "\t acc:" + str(
            acc_mean) + "\t AUC:" + str(AUC_mean)
        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_test = '\t val rmse' + str(val_rmse) + '\t test rmse:' + str(test_rmse) + '\t fair 50:'+str(fairness50)+'\t fair all:'+str(fairness_all)
        print(str_print_test)
        result_file.write(str_print_test)
        result_file.write('\n')
        result_file.flush()

def baseline_faircf_user():
    print('------------training processing--------------')
    time1 = time.time()
    with open('./preprocessed/last_fm_preprocessed.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
    # pdb.set_trace()
    time2 = time.time()
    print('Load complete, use time: '+str(time2-time1))

    train_pair_tmp = np.array(train_pair)
    print(train_pair_tmp.shape)
    avg_rating = np.mean(train_pair_tmp[:,2])
    print("avg:"+str(avg_rating))
    avg_rating = np.array(avg_rating)
    print(avg_rating)

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    train_dataset = TripletUniformPair(train_pair, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=65536, shuffle=False)
    val_dataset = TripletUniformPair(val_pair, val_set_len)
    val_loader = DataLoader(val_dataset, batch_size=65536, shuffle=False)

    model = BPR_bias_faircf_user(user_size, item_size, factor_num, avg_rating).cuda()
    fairD_gender_user = GenderDiscriminator(factor_num, gender_data).cuda()
    optimizer_fairD_gender_user = torch.optim.Adam(fairD_gender_user.parameters(), lr=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    log_path = path_save_log_base + '/date_1204_faircf_user_pmf' + '.txt'
    print(log_path)
    result_file = open(log_path, 'a')
    result_file.write('\n\n')
    result_file.write('train MF on lastfm')

    model.train()
    fairD_gender_user.train()
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_2.pt"))
    fairD_gender_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_2.pt"))

    for epoch in range(200):
        model.train()
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum6 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, i, r) in enumerate(train_loader):
            fairD_gender_user.eval()

            u = u.cuda()
            i = i.cuda()
            r = r.cuda()
            optimizer.zero_grad()
            task_loss, recommend_loss, d_loss, r_loss = model(u, i, r, Discrminator=fairD_gender_user)
            task_loss.backward()
            optimizer.step()
            train_loss_sum.append(recommend_loss.item())
            train_loss_sum2.append(task_loss.item())
            train_loss_sum3.append(d_loss.item())


            D_steps = 20
            for _ in range(D_steps):
                fairD_gender_user.train()
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
                optimizer_fairD_gender_user.zero_grad()
                l_penalty_user = fairD_gender_user(user_embedding.detach(), u, True)
                l_penalty_user.backward()
                optimizer_fairD_gender_user.step()
                if _ == 0:
                    train_loss_sum6.append(l_penalty_user.item())

            with torch.no_grad():
                preds_list4, labels_list4, probs_list4 = [], [], []
                age_correct = []
                user_embedding, item_embedding = model(u, i, r, return_batch_embedding=True)
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

        ####################
        #    save model    #
        ####################
        # 21 - 149
        PATH_model=path_save_model_base+'/cfu_'+str(epoch)+'.pt'  # -15 -100
        str_save = 'save this epoch\'s model to this path:' + PATH_model + '\n'
        result_file.write(str_save)
        torch.save(model.state_dict(), PATH_model)
        model.eval()
        ###############
        #  eval part  #
        ###############
        fairness, nips2017 = 0, 0
        with torch.no_grad():
            val_rmse = 0
            for idx, (u, i, r) in enumerate(val_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                val_rmse += rmse.item()
                # pdb.set_trace()
            val_rmse = np.sqrt(val_rmse / val_set_len)
            test_rmse = 0
            for idx, (u, i, r) in enumerate(test_loader):
                u, i, r = u.cuda(), i.cuda(), r.cuda()
                rmse = model.predict(u, i, r)
                test_rmse += rmse.item()
            test_rmse = np.sqrt(test_rmse / test_set_len)

            users_embedding, items_embedding, user_bias_tmp, item_bias_tmp = model.predict(0, 0, 0, return_e=True)
            user_e = users_embedding.cpu().detach().numpy().astype(np.float32)
            item_e = items_embedding.cpu().detach().numpy().astype(np.float32)
            user_bias_tmp = user_bias_tmp.cpu().detach().numpy().astype(np.float32)
            item_bias_tmp = item_bias_tmp.cpu().detach().numpy().astype(np.float32)
            user_e = user_e[:10000]
            item_e = item_e[:2500]
            user_bias_tmp = user_bias_tmp[:10000]
            item_bias_tmp = item_bias_tmp[:, :2500]
            gender_data = gender_data[:10000]
            fairness_all, fairness50 = fair_K_lastfm_bias(user_e, item_e, user_bias_tmp, item_bias_tmp, gender_data,
                                                          train_user_list)

        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss6 = round(np.mean(train_loss_sum6[:-1]), 4)
        AUC_mean = round(np.mean(AUC_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time

        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t task loss:' + str(
            train_loss) + "\t all loss  " + str(train_loss2) + "\t user_cls_loss:" +str(
            train_loss3)+ '\t cls_loss2:'+str(train_loss6)  + "\t acc:" + str(
            acc_mean) + "\t AUC:" + str(AUC_mean)
        print(' train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        str_print_test = '\t val rmse' + str(val_rmse) + '\t test rmse:' + str(test_rmse) + '\t fair 50:'+str(fairness50)+'\t fair all:'+str(fairness_all)
        print(str_print_test)
        result_file.write(str_print_test)
        result_file.write('\n')
        result_file.flush()

if __name__ == '__main__':
    # test_MF()
    baseline_faircf()
    # train_newDisc()