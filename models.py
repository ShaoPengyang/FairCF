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
import torch.optim as optim
from torch.nn.init import xavier_normal, xavier_uniform
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score,roc_curve,auc
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
import torch.autograd as autograd
import pdb
lamada = 0.001

# class Filter(nn.Module):
#     def __init__(self,factor_num,shared=True):
#         super(Filter, self).__init__()
#         self.criterion = nn.MSELoss()
#         self.rmse = nn.MSELoss()
#         self.embed_dim = factor_num
#         self.user_filter = nn.Sequential(
#             nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
#             nn.LeakyReLU(0.1),
#             nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
#             nn.LeakyReLU(0.1),
#             nn.Linear(self.embed_dim, self.embed_dim, bias=True),
#         )
#         self.share = shared
#
#     def forward(self,user0,user_embedding,item0,item_embedding,ratings,return_batch_embedding = False,Discrminator=None):
#         user_embedding = self.user_filter(user_embedding)
#         if self.share == True:
#             item_embedding = self.user_filter(item_embedding)
#         if return_batch_embedding == True:
#             return user_embedding,item_embedding
#         ratings = ratings.float()
#         prediction_i = (user_embedding * item_embedding).sum(dim=-1)
#         l2_regulization = lamada * (user_embedding ** 2).mean() + lamada * (item_embedding ** 2).mean()
#         loss2 = self.criterion(prediction_i, ratings)
#         loss = loss2+l2_regulization
#         l_penalty_1 = 0
#         if Discrminator is not None:
#             l_penalty_1 = Discrminator(user_embedding,user0, True)
#             # p lamada mf 50 20/25 l2 0.001
#             # loss = loss  - 50 * l_penalty_1
#
#             # gcn   -20
#             loss = loss  - 20 * l_penalty_1
#         return loss, loss2, l_penalty_1 , l2_regulization
#
#     def predict(self,user0,user_embedding,item0,item_embedding,ratings,return_batch_embedding = False):
#         user_embedding = self.user_filter(user_embedding)
#         if self.share == True:
#             item_embedding = self.user_filter(item_embedding)
#         if return_batch_embedding == True:
#             return user_embedding, item_embedding
#         ratings = ratings.float()
#         prediction_i = (user_embedding * item_embedding).sum(dim=-1)
#         rmse = self.rmse(prediction_i, ratings)
#         return rmse
#
# class Filter_2adv(nn.Module):
#     def __init__(self,factor_num):
#         super(Filter_2adv, self).__init__()
#         self.criterion = nn.MSELoss()
#         self.rmse = nn.MSELoss()
#         self.embed_dim = factor_num
#         self.user_filter = nn.Sequential(
#             nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
#             nn.LeakyReLU(0.1),
#             nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
#             nn.LeakyReLU(0.1),
#             nn.Linear(self.embed_dim, self.embed_dim, bias=True),
#         )
#
#     def forward(self,user0,user_embedding,item0,item_embedding,ratings,return_batch_embedding = False,Discrminator=None,Regression=None):
#         user_embedding = self.user_filter(user_embedding)
#         item_embedding = self.user_filter(item_embedding)
#
#         if return_batch_embedding == True:
#             return user_embedding,item_embedding
#         ratings = ratings.float()
#         prediction_i = (user_embedding * item_embedding).sum(dim=-1)
#         l2_regulization = lamada * (user_embedding ** 2).mean() + lamada * (item_embedding ** 2).mean()
#         loss2 = self.criterion(prediction_i, ratings)
#         loss = loss2 + l2_regulization
#         l_penalty_1,l_penalty_2=0,0
#         if Discrminator is not None:
#             l_penalty_1 = Discrminator(user_embedding,user0, True)
#             l_penalty_2 = Regression(item_embedding,item0,True)
#             # unshare 20  shared 35
#             # MF 50 100
#             # GCN 20 40
#             loss = loss - 20 * l_penalty_1 - 40*l_penalty_2
#         return loss, loss2, l_penalty_1, l_penalty_2
#
#     def predict(self,user0,user_embedding,item0,item_embedding,ratings,return_batch_embedding = False):
#         user_embedding = self.user_filter(user_embedding)
#         item_embedding = self.user_filter(item_embedding)
#
#         if return_batch_embedding == True:
#             return user_embedding, item_embedding
#         ratings = ratings.float()
#         prediction_i = (user_embedding * item_embedding).sum(dim=-1)
#         rmse = self.rmse(prediction_i, ratings)
#         return rmse

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)


    def forward(self, user0, item_i0, ratings, return_batch_embedding = False):
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        if return_batch_embedding == True:
            return user,item_i
        else:
            return loss, loss2,l2_regulization

    def predict(self,user0,item_i0,ratings,return_e = False,return_batch_embedding=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight

            return users_embedding, items_embedding

        if return_batch_embedding == True:
            users_embedding = self.embed_user(user0)
            items_embedding = self.embed_item(item_i0)
            return users_embedding, items_embedding

        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)

        # user = self.user_filter(user_)
        # item_i = self.user_filter(item_i)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_bias(nn.Module):
    def __init__(self, user_size, item_size, factor_num,avg_rating):
        super(BPR_bias, self).__init__()
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.embed_user = nn.Embedding(user_size, factor_num)
        self.embed_item = nn.Embedding(item_size, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_bias = nn.Embedding(user_size, 1)
        self.item_bias = nn.Embedding(item_size, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings, return_batch_embedding = False):
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        task_loss = self.criterion(prediction_i, ratings)
        loss = task_loss + l2_regulization

        if return_batch_embedding == True:
            return user, item_i, user_bias, item_bias
        else:
            return loss, task_loss, l2_regulization

    def predict(self,user0,item_i0,ratings,return_e = False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            # pdb.set_trace()
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_biasNN(nn.Module):
    def __init__(self, user_size, item_size, factor_num,avg_rating):
        super(BPR_biasNN, self).__init__()
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.rmse = nn.MSELoss(reduction='mean')
        self.embed_user = nn.Embedding(user_size, factor_num)
        self.embed_item = nn.Embedding(item_size, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_bias = nn.Embedding(user_size, 1)
        self.item_bias = nn.Embedding(item_size, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            # nn.LeakyReLU(0.1),
            # nn.Linear(self.embed_dim * 2, self.embed_dim*2, bias=True),
            # nn.LeakyReLU(0.1),
            # nn.Linear(self.embed_dim*2, self.embed_dim, bias=True),
            # nn.LeakyReLU(0.1),
        )

    def forward(self, user0, item_i0, ratings, return_batch_embedding = False):
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        task_loss = self.criterion(prediction_i, ratings)
        loss = task_loss + l2_regulization

        if return_batch_embedding == True:
            return user, item_i, user_bias, item_bias
        else:
            return loss, task_loss, l2_regulization

    def predict(self,user0,item_i0,ratings,return_e = False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            items_embedding = self.user_filter(items_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_2_filter_bias(nn.Module):
    def __init__(self, user_num, item_num, factor_num,avg_rating):
        super(BPR_2_filter_bias, self).__init__()
        self.rmse = nn.MSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim*2,  bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embed_dim*2, self.embed_dim, bias=True)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None,Regression = None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0,0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            # for gender
            # loss = loss - 10 *l_penalty_1
            # for age
            loss = loss -  l_penalty_1
        if Regression is not None:
            l_penalty_2 = Regression(item_i, item_i0, True)

            loss = loss - 20 * l_penalty_2

        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            items_embedding = self.user_filter(items_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_icml_bias(nn.Module):
    def __init__(self, user_num, item_num, factor_num,avg_rating):
        super(BPR_icml_bias, self).__init__()
        self.rmse = nn.MSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0, 0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            loss = loss - 10 * l_penalty_1
        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_bias_faircf_user(nn.Module):
    def __init__(self, user_num, item_num, factor_num,avg_rating):
        super(BPR_bias_faircf_user, self).__init__()
        self.rmse = nn.MSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0, 0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            loss = loss - 10 * l_penalty_1
        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            items_embedding = self.user_filter(items_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)

        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_user_filter(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR_user_filter, self).__init__()
        self.criterion = nn.MSELoss()
        # self.rmse = nn.MSELoss(reduction='mean')
        self.rmse = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        # filter
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None):
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        # filter only use
        if Discrminator is not None:
            user =  self.user_filter(user)
            item_i = self.user_filter(item_i)

        if return_batch_embedding == True:
            return user,item_i

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1 = 0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            # 10 15 20
            loss = loss - 10 * l_penalty_1

        return loss, loss2, l_penalty_1 , l2_regulization

    def predict(self,user0,item_i0,ratings,return_e = False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight

            users_embedding = self.user_filter(users_embedding)
            items_embedding = self.user_filter(items_embedding)

            return users_embedding, items_embedding

        user_ = self.embed_user(user0)
        item_i_ = self.embed_item(item_i0)

        user = self.user_filter(user_)
        item_i = self.user_filter(item_i_)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_user_filter_low(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR_user_filter_low, self).__init__()
        self.criterion = nn.MSELoss()
        # self.rmse = nn.MSELoss(reduction='mean')
        self.rmse = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        # filter
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None):
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        if Discrminator is not None:
            user =  self.user_filter(user)

        if return_batch_embedding == True:
            return user,item_i

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1 = 0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            # 10 15 20
            loss = loss - 15 * l_penalty_1

        return loss, loss2, l_penalty_1 , l2_regulization

    def predict(self,user0,item_i0,ratings,return_e = False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight

            users_embedding = self.user_filter(users_embedding)

            return users_embedding, items_embedding

        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)

        user = self.user_filter(user)

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_2_filter(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR_2_filter, self).__init__()
        self.rmse = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None,Regression = None):
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        if Discrminator is not None:
            user = self.user_filter(user)
            item_i = self.user_filter(item_i)

        if return_batch_embedding == True:
            return user,item_i

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0,0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            loss = loss - 10 * l_penalty_1
        if Regression is not None:
            l_penalty_2 = Regression(item_i, item_i0, True)
            loss = loss - 50 * l_penalty_2

        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            items_embedding = self.user_filter(items_embedding)
            return users_embedding, items_embedding

        user_ = self.embed_user(user0)
        item_i_ = self.embed_item(item_i0)

        user = self.user_filter(user_)
        item_i = self.user_filter(item_i_)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GenderDiscriminator(nn.Module):
    def __init__(self, embed_dim, inds):
        super(GenderDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.users_sensitive = inds
        self.out_dim = 1
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim/2), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 4), int(self.embed_dim / 8), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 8), self.out_dim, bias = True)
        )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = self.sigmoid(scores)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        if return_loss:
            loss1 = self.criterion(output.squeeze(), A_labels)
            return loss1
        else:
            return output.squeeze(), A_labels

    def predict(self, ents_emb, ents, return_loss=True, return_preds=False, cpu_tensor=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.sigmoid(scores)
            A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
        if return_preds:
            return output.squeeze(), A_labels, preds
        elif return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(), A_labels

class GenderRegression(nn.Module):
    def __init__(self, embed_dim, inds):
        super(GenderRegression, self).__init__()
        self.rmse = nn.MSELoss()
        self.embed_dim = int(embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.users_sensitive = inds
        self.out_dim = 2
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 4), int(self.embed_dim / 8), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 8), self.out_dim, bias=True)
        )

    def forward(self, ents_emb, ents, return_loss=False):
        output = self.net(ents_emb)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        if return_loss:
            loss1 = self.criterion(output.squeeze(), A_labels)
            return loss1
        else:
            return output.squeeze(), A_labels

    def predict(self,ents_emb, ents):
        output = self.net(ents_emb)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        rmse = self.rmse(output.squeeze(), A_labels)
        return rmse

class GenderRegression_onedim(nn.Module):
    def __init__(self, embed_dim, inds):
        super(GenderRegression_onedim, self).__init__()
        self.rmse = nn.MSELoss()
        self.embed_dim = int(embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.users_sensitive = inds
        self.out_dim = 1
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 4), int(self.embed_dim / 8), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 8), self.out_dim, bias=True)
        )

    def forward(self, ents_emb, ents, return_loss=False):
        output = self.net(ents_emb)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        if return_loss:
            loss1 = self.criterion(output.squeeze(), A_labels)
            loss1 = torch.sqrt(loss1)
            return loss1
        else:
            return output.squeeze(), A_labels

    def predict(self,ents_emb, ents):
        output = self.net(ents_emb)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        rmse = self.rmse(output.squeeze(), A_labels)
        return rmse

class KBDataset(Dataset):
    def __init__(self,data_split,prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        np.random.shuffle(data)
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()

class TripletUniformPair(Dataset):
    def __init__(self, user_list, set_len):
        super(TripletUniformPair, self).__init__()
        self.user_list = user_list
        self.set_len = set_len

    def __getitem__(self, idx):
        uij = self.user_list
        u = uij[idx][0]
        i = uij[idx][1]
        r = uij[idx][2]
        return u, i, r

    def __len__(self):
        return self.set_len

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

class GCN_bias(nn.Module):
    def __init__(self, user_num, item_num,  factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,avg_rating):
        super(GCN_bias, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)


        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding+gcn1_items_embedding+gcn2_items_embedding

        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding,gcn4_users_embedding),-1)  # +gcn4_users_embedding
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding,gcn4_items_embedding),-1)  # +gcn4_items_embedding#

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        return loss, loss2, l2_regulization

    def predict(self,  user0, item_i0,ratings,return_e = False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)))  # *2. + items_embedding


        gcn_users_embedding = users_embedding + gcn1_users_embedding + gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding,gcn3_users_embedding,gcn4_users_embedding),-1)
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding,gcn3_items_embedding,gcn4_items_embedding), -1)

        if return_e == True:
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return gcn_users_embedding, gcn_items_embedding, user_bias, item_bias.squeeze(1).unsqueeze(0)

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)

        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_icml_2019(nn.Module):
    def __init__(self, user_num, item_num,  factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,avg_rating):
        super(GCN_icml_2019, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings, Disc=None, return_batch_embedding=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding+gcn1_items_embedding+gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        # gcn_items_embedding = self.user_filter(gcn_items_embedding)


        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        if return_batch_embedding ==True:
            return user,item_i
        # # pdb.set_trace()
        ratings = ratings.float()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1, l_penalty_2 = 0, 0
        if Disc is not None:
            l_penalty_1 = Disc(user, user0, True)
            loss = loss - 10 * l_penalty_1
        return loss, loss2,l_penalty_1,l_penalty_2

    def predict(self,  user0, item_i0,ratings,return_e = False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        # gcn_items_embedding = self.user_filter(gcn_items_embedding)


        if return_e == True:
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return gcn_users_embedding,gcn_items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        ratings = ratings.float()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_bias_faircf_user(nn.Module):
    def __init__(self, user_num, item_num,  factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,avg_rating):
        super(GCN_bias_faircf_user, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings, Disc=None, return_batch_embedding=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding+gcn1_items_embedding+gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        gcn_items_embedding = self.user_filter(gcn_items_embedding)


        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        if return_batch_embedding ==True:
            return user,item_i
        # # pdb.set_trace()
        ratings = ratings.float()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1, l_penalty_2 = 0, 0
        if Disc is not None:
            l_penalty_1 = Disc(user, user0, True)
            loss = loss - 10*l_penalty_1
        return loss, loss2,l_penalty_1,l_penalty_2

    def predict(self,  user0, item_i0,ratings,return_e = False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        gcn_items_embedding = self.user_filter(gcn_items_embedding)


        if return_e == True:
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return gcn_users_embedding,gcn_items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        ratings = ratings.float()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_bias_faircf(nn.Module):
    def __init__(self, user_num, item_num,  factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,avg_rating):
        super(GCN_bias_faircf, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,Reg = None,Disc=None,return_batch_embedding=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding+gcn1_items_embedding+gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        gcn_items_embedding = self.user_filter(gcn_items_embedding)


        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        if return_batch_embedding ==True:
            return user,item_i
        # # pdb.set_trace()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1, l_penalty_2 = 0, 0
        if Disc is not None:
            l_penalty_1 = Disc(user, user0, True)
            # gender is 10 10
            # for age
            loss = loss - 1 *l_penalty_1
        if Reg is not None:
            l_penalty_2 = Reg(item_i, item_i0, True)
            loss = loss - 40*l_penalty_2
        return loss, loss2,l_penalty_1,l_penalty_2

    def predict(self,  user0, item_i0,ratings,return_e = False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        gcn_items_embedding = self.user_filter(gcn_items_embedding)

        if return_e == True:
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return gcn_users_embedding,gcn_items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_user_filter_low(nn.Module):
    def __init__(self, user_num, item_num,  factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,shared=True):
        super(GCN_user_filter_low, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss()
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.share=shared

    def forward(self, user0, item_i0, ratings,Disc=None,return_batch_embedding=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding+gcn1_items_embedding+gcn2_items_embedding

        if Disc is not None:
            gcn_users_embedding = self.user_filter(gcn_users_embedding)
            if self.share == True:
                gcn_items_embedding = self.user_filter(gcn_items_embedding)

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)


        if return_batch_embedding ==True:
            return user,item_i
        # # pdb.set_trace()
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2
        l_penalty_1 = 0
        if Disc is not None:
            l_penalty_1 = Disc(user, user0, True)
            # shared 20
            loss = loss - 20*l_penalty_1
        return loss, loss2,l_penalty_1,l2_regulization

    def predict(self,  user0, item_i0,ratings,return_e = False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)

        if self.share == True:
            gcn_items_embedding = self.user_filter(gcn_items_embedding)

        if return_e == True:
            return gcn_users_embedding,gcn_items_embedding

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1)
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class AgeDiscriminator(nn.Module):
    def __init__(self, embed_dim, inds):
        super(AgeDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.softmax = nn.Softmax()
        self.criterion = nn.CrossEntropyLoss()
        self.users_sensitive = inds
        self.out_dim = 7
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim/2), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 4), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True)
        )

    def forward(self, ents_emb, ents, return_loss=False):
        output = self.net(ents_emb)
        # output = self.sigmoid(scores)
        A_labels = Variable(torch.cuda.LongTensor(self.users_sensitive[ents.cpu()]))
        if return_loss:
            loss1 = self.criterion(output.squeeze(), A_labels)
            return loss1
        else:
            return output.squeeze(), A_labels

    def predict(self, ents_emb, ents, return_loss=True, return_preds=False, cpu_tensor=False):
        with torch.no_grad():
            output = self.net(ents_emb)
            # pdb.set_trace()
            A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
            # preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
            preds = None
        if return_preds:
            output = self.softmax(output)
            return output.squeeze(), A_labels, preds
        elif return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            output = self.softmax(output)
            return output.squeeze(), A_labels

class AgerRegression(nn.Module):
    def __init__(self, embed_dim, inds):
        super(AgerRegression, self).__init__()
        self.rmse = nn.MSELoss()
        self.embed_dim = int(embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.users_sensitive = inds
        self.out_dim = 7
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim / 4), self.out_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, ents_emb, ents, return_loss=False):
        output = self.net(ents_emb)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        if return_loss:
            loss1 = self.criterion(output.squeeze(), A_labels)
            return loss1
        else:
            return output.squeeze(), A_labels

    def predict(self,ents_emb, ents):
        output = self.net(ents_emb)
        A_labels = Variable(torch.cuda.FloatTensor(self.users_sensitive[ents.cpu()]))
        rmse = self.rmse(output.squeeze(), A_labels)
        return rmse

class BPR_compostional(nn.Module):
    def __init__(self, user_num, item_num, factor_num,avg_rating):
        super(BPR_compostional, self).__init__()
        self.rmse = nn.MSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim*2,  bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embed_dim*2, self.embed_dim, bias=True)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings, return_batch_embedding=False, GenDisc=None, GenReg = None,
                AgeDisc=None, AgeReg = None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0,0

        if GenDisc is not None:
            l_penalty_1 = GenDisc(user, user0, True)
            loss = loss - 10 *l_penalty_1

        if GenReg is not None:
            l_penalty_2 = GenReg(item_i, item_i0, True)
            loss = loss - 20 * l_penalty_2

        if AgeDisc is not None:
            l_penalty_3 = AgeDisc(user, user0, True)
            loss = loss - 1 * l_penalty_3

        if AgeReg is not None:
            l_penalty_4 = AgeReg(item_i, item_i0, True)
            loss = loss - 20 * l_penalty_4
        return loss, loss2, l_penalty_1,l_penalty_2,l_penalty_3,l_penalty_4

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            items_embedding = self.user_filter(items_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        item_i = self.user_filter(item_i)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_icml_age(nn.Module):
    def __init__(self, user_num, item_num, factor_num,avg_rating):
        super(BPR_icml_age, self).__init__()
        self.rmse = nn.MSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, Discrminator=None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0, 0
        if Discrminator is not None:
            l_penalty_1 = Discrminator(user, user0, True)
            loss = loss - l_penalty_1
        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class BPR_icml_com(nn.Module):
    def __init__(self, user_num, item_num, factor_num,avg_rating):
        super(BPR_icml_com, self).__init__()
        self.rmse = nn.MSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        self.embed_dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, GenDisc=None,AgeDisc=None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0, 0
        if GenDisc is not None:
            l_penalty_1 = GenDisc(user, user0, True)
            loss = loss - 10 * l_penalty_1
        if AgeDisc is not None:
            l_penalty_2 = AgeDisc(user, user0, True)
            loss = loss -  l_penalty_2
        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_compostional(nn.Module):
    def __init__(self, user_num, item_num,  factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,avg_rating):
        super(GCN_compostional, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings, return_batch_embedding=False, GenDisc=None, GenReg = None,
                AgeDisc=None, AgeReg = None):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding+gcn1_items_embedding+gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        gcn_items_embedding = self.user_filter(gcn_items_embedding)


        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        if return_batch_embedding ==True:
            return user,item_i
        # # pdb.set_trace()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1, l_penalty_2 = 0, 0
        if GenDisc is not None:
            l_penalty_1 = GenDisc(user, user0, True)
            # gender is 10 10
            # for age
            loss = loss - 10 *l_penalty_1
        if GenReg is not None:
            l_penalty_2 = GenReg(item_i, item_i0, True)
            loss = loss - 10*l_penalty_2

        if AgeDisc is not None:
            l_penalty_3 = AgeDisc(user, user0, True)
            loss = loss - 1 * l_penalty_3

        if AgeReg is not None:
            l_penalty_4 = AgeReg(item_i, item_i0, True)
            loss = loss - 40* l_penalty_4
        return loss, loss2, l_penalty_1, l_penalty_2, l_penalty_3, l_penalty_4

    def predict(self,  user0, item_i0,ratings,return_e = False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding+gcn1_users_embedding+gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        gcn_items_embedding = self.user_filter(gcn_items_embedding)

        if return_e == True:
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return gcn_users_embedding,gcn_items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_icml_age(nn.Module):
    def __init__(self, user_num, item_num, factor_num, user_item_matrix, item_user_matrix, d_i_train, d_j_train,
                 avg_rating):
        super(GCN_icml_age, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings, Disc=None, return_batch_embedding=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu(
            (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
                self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu(
            (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
                self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding + gcn1_users_embedding + gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        # gcn_items_embedding = self.user_filter(gcn_items_embedding)

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        if return_batch_embedding == True:
            return user, item_i
        # # pdb.set_trace()
        ratings = ratings.float()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1, l_penalty_2 = 0, 0
        if Disc is not None:
            l_penalty_1 = Disc(user, user0, True)
            loss = loss - 1 * l_penalty_1
        return loss, loss2, l_penalty_1, l_penalty_2

    def predict(self, user0, item_i0, ratings, return_e=False):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight

        gcn1_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train)))  # *2. #+ users_embedding
        gcn1_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train)))  # *2. #+ items_embedding

        gcn2_users_embedding = F.relu((torch.sparse.mm(self.user_item_matrix,
                                                       gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train)))  # *2. + users_embedding
        gcn2_items_embedding = F.relu((torch.sparse.mm(self.item_user_matrix,
                                                       gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train)))  # *2. + items_embedding

        gcn_users_embedding = users_embedding + gcn1_users_embedding + gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding

        gcn_users_embedding = self.user_filter(gcn_users_embedding)
        # gcn_items_embedding = self.user_filter(gcn_items_embedding)

        if return_e == True:
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return gcn_users_embedding, gcn_items_embedding, user_bias, item_bias.squeeze(1).unsqueeze(0)

        user = F.embedding(user0, gcn_users_embedding)
        item_i = F.embedding(item_i0, gcn_items_embedding)
        # # pdb.set_trace()
        ratings = ratings.float()
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse

class GCN_icml_com(nn.Module):
    def __init__(self, user_num, item_num, factor_num, user_item_matrix, item_user_matrix, d_i_train, d_j_train,
                 avg_rating):
        super(GCN_icml_com, self).__init__()
        '''
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        '''
        self.embed_dim = factor_num
        self.criterion = nn.MSELoss()
        self.rmse = nn.MSELoss(reduction='mean')
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        # self.add = nn.Linear(3, 1, bias=True)
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.user_filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim, bias=True),
            nn.LeakyReLU(0.1)
        )
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        self.avg_rating = torch.cuda.FloatTensor(avg_rating)

    def forward(self, user0, item_i0, ratings,return_batch_embedding=False, GenDisc=None,AgeDisc=None):
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        if return_batch_embedding == True:
            return user,item_i
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.criterion(prediction_i, ratings)
        loss = loss2 + l2_regulization
        l_penalty_1,l_penalty_2 = 0, 0
        if GenDisc is not None:
            l_penalty_1 = GenDisc(user, user0, True)
            loss = loss - 10 * l_penalty_1
        if AgeDisc is not None:
            l_penalty_2 = AgeDisc(user, user0, True)
            loss = loss -  l_penalty_2
        return loss, loss2, l_penalty_1,l_penalty_2

    def predict(self, user0, item_i0, ratings,return_e=False):
        if return_e == True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            users_embedding = self.user_filter(users_embedding)
            user_bias = self.user_bias.weight
            item_bias = self.item_bias.weight
            return users_embedding, items_embedding,user_bias,item_bias.squeeze(1).unsqueeze(0)
        user_bias = self.user_bias(user0)
        item_bias = self.item_bias(item_i0)
        user = self.embed_user(user0)
        item_i = self.embed_item(item_i0)
        user = self.user_filter(user)
        ratings = ratings.float()
        prediction_i = (user * item_i).sum(dim=-1) + user_bias.squeeze(1) + item_bias.squeeze(1) + self.avg_rating
        rmse = self.rmse(prediction_i, ratings)
        return rmse