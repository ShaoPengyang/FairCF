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
from model import *
from evaluate import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="change gpuid")
parser.add_argument("-g", "--gpu-id", help="choose which gpu to use", type=str, default=str(3))
parser.add_argument("-t", "--times", help="choose times", type=str, default='0')
# 0 epoch 41 no adversary
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
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
batch_size = 4096*32
lamada = 0.01

def evaluate_fair():
    print('------------eval processing--------------')
    time1 = time.time()
    # Load preprocess data
    with open('./preprocessed/last_fm_gcn.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['all_user_list'], dataset[
            'test_user_list']
        all_user_list = dataset['all_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
    time2 = time.time()
    print('Load complete, use time: '+str(time2-time1))

    train_set_len, test_set_len, val_set_len=len(train_pair), len(test_pair), len(val_pair)
    test_dataset = TripletUniformPair(test_pair, test_set_len)
    test_loader = DataLoader(test_dataset, batch_size=test_set_len, shuffle=False)
    model = BPR(user_size, item_size, factor_num).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    result_file = open(path_save_log_base + '/results_fair_no_adv.txt', 'a')

    # with open('./nips.pickle', 'rb') as f:
    #     dataset = pickle.load(f)
    #     weights = dataset['weights']
    #     test_item_index = dataset['test_item_index']
    #
    # test_item_num = 140437
    # tensor_test_weights = torch.cuda.FloatTensor(weights)

    for epoch in range(15):
        PATH_model=path_save_model_base+'/'+str(epoch)+'_1.pt'
        model.train()
        model.load_state_dict(torch.load(PATH_model))
        model.eval()
        with torch.no_grad():
            # for idx, (u, i, r) in enumerate(test_loader):
            #     u, i, r = u.cuda(), i.cuda(), r.cuda()
            #     U_val = model.nips2017(u,i,r,tensor_test_weights,test_item_num,test_item_index)

            fairness_K50 = model.fairness(gender_data,50,train_user_list, val_user_list)
            # user_6040_embedding = user_6040_embedding.cpu().detach().numpy().astype('float16')
            # item_3706_embedding = item_3706_embedding.cpu().detach().numpy().astype('float16')
            # fairness_K50 = fair_K_lastfm(user_6040_embedding, item_3706_embedding, 50, test_user_list, all_user_list,gender_data)

        str_print_train = "epoch:" + str(epoch)+'\t fairness:'+str(fairness_K50)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n\n')
        result_file.flush()

if __name__ == '__main__':
    evaluate_fair()
