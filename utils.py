import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import ipdb
import dill as pkl
import time
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# load weights
def get_updateModel(model, sasrec_model_path):
#     ipdb.set_trace()
    pretrained_dict_target = torch.load(sasrec_model_path, map_location='cpu') # 68
    model_dict = model.state_dict() # 68
    shared_dict_target = {k: v for k, v in pretrained_dict_target.items() if k.startswith('sasrec_embedding')}# 28
    model_dict.update(shared_dict_target)
    
    print("Load the length of target is:", len(shared_dict_target.keys()))

    model.load_state_dict(model_dict)
    return model


def generate_file(args, user_list, user_id_train, user_id_valid, user_id_test, item_entries, itemnum):
    print('process user file!')
    seq_id = np.zeros([len(user_list)+1, args.maxlen], dtype=np.int32)        
    pos_id = np.zeros([len(user_list)+1, args.maxlen], dtype=np.int32)
    seq_id_test = np.zeros([len(user_list)+1, args.maxlen], dtype=np.int32)
    pos_id_test = np.zeros([len(user_list)+1, 1], dtype=np.int32)
    sample_pool_test = np.zeros([len(user_list)+1, itemnum-1], dtype=np.int32)
    for u in user_list:
        nxt_id = user_id_train[u][-1] # # 最后一个交互的物品
        idx = args.maxlen - 1 #49
        for i in reversed(range(0, len(user_id_train[u][:-1]))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_id[u,idx] = user_id_train[u][i]
            pos_id[u,idx] = nxt_id
            nxt_id = user_id_train[u][i]
            idx -= 1
            if idx == -1: 
                break
                
        idx_test = args.maxlen - 1 #49
        seq_id_test[u,idx_test] = user_id_valid[u][0]

        idx_test -= 1  
        for i in reversed(range(0, len(user_id_train[u]))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_id_test[u,idx_test] = user_id_train[u][i]
            # 为什么要设定不等于0？是为了保证当序列长度没到达maxlen时，正样本序列会补充为0，那么构成的负样本序列也应该是0
            idx_test -= 1
            if idx_test == -1: 
                break
                
        sample_pool_test[u] = np.setdiff1d(item_entries, user_id_test[u][0])
        pos_id_test[u] = user_id_test[u][0]
                
    return seq_id, pos_id, seq_id_test,  sample_pool_test,  pos_id_test


            
# source:book----range[1,interval+1);target:movie[interval+1, itemnum + 1)
def sample_function(random_min, random_max, user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, user_list, item_categoory_dict, seq_id_fortrain, pos_id_fortrain, w_min, w_max, reweight_version, overlap_version, batch_size, maxlen, sample_ratio, result_queue):        
    
    def sample():      
        user = np.random.randint(1, usernum + 1)
        while user not in user_list:
            user = np.random.randint(1, usernum + 1)
        
        user_id_train_u = np.array(user_id_train[user])
        seq_id = seq_id_fortrain[user]
        pos_id = pos_id_fortrain[user]
        neg_id = np.zeros([sample_ratio, maxlen], dtype=np.int32)
        nxt_id = user_id_train_u[-1] # # 最后一个交互的物品
        idx = maxlen - 1 #49
        ts_id = set(user_id_train_u) # a set

        for i in reversed(range(0, len(user_id_train_u[:-1]))): 
            if nxt_id != 0: 
                for j in range(0,sample_ratio):
                    neg_id[j, idx] = random_neq(1, itemnum+1, ts_id)
            nxt_id = user_id_train_u[i]
            idx -= 1
            if idx == -1: break

        return (user, seq_id, pos_id, neg_id)

    
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))    
            
class WarpSampler(object):
    def __init__(self, random_min, random_max, user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, user_list, args_DM, item_categoory_dict, seq_id_fortrain, pos_id_fortrain, w_min=0.1, w_max=1.0, batch_size=64, maxlen=10, n_workers=1, sample_ratio=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(random_min, random_max, user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, user_list, item_categoory_dict, seq_id_fortrain,  pos_id_fortrain, w_min, w_max, args_DM.reweight_version, args_DM.overlap_version, batch_size, maxlen, sample_ratio, self.result_queue )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()   

            
def data_partition(fname, dataset_name, maxlen):
    usernum = 0
    itemnum = 0
    user_train = {}
    user_valid = {}
    user_test = {}
    interval = 0
#     ipdb.set_trace()
    # assume user/item index starting from 1
    if dataset_name == 'Home':
        with open('./Datasets/Home/log_file_final.pkl', 'rb') as f:
            log_file_final = pkl.load(f)
        with open('./Datasets/Home/log_timestep_final.pkl', 'rb') as f:
            log_timestep_final = pkl.load(f)
        with open('./Datasets/Home/log_category_final.pkl', 'rb') as f:
            log_category_final = pkl.load(f)
            
        with open('./Datasets/Home/item_index_id_dict.pkl', 'rb') as f:
            item_index_id_dict = pkl.load(f)
        with open('./Datasets/Home/item_id_index_dict.pkl', 'rb') as f:
            item_id_index_dict = pkl.load(f)
            
        with open('./Datasets/Home/category_item_dict.pkl', 'rb') as f:
            category_item_dict = pkl.load(f)
        with open('./Datasets/Home/item_categoory_dict.pkl', 'rb') as f:
            item_category_dict= pkl.load(f)
            
        usernum = len(log_file_final.keys()) # 7176
        itemnum = len(item_index_id_dict.keys()) # 7506
        catenum = len(category_item_dict.keys()) # 407 
    elif dataset_name == 'Electronic':
        with open('./Datasets/Electronic/log_file_final.pkl', 'rb') as f:
            log_file_final = pkl.load(f)
        with open('./Datasets/Electronic/log_timestep_final.pkl', 'rb') as f:
            log_timestep_final = pkl.load(f)
        with open('./Datasets/Electronic/log_category_final.pkl', 'rb') as f:
            log_category_final = pkl.load(f)
            
        with open('./Datasets/Electronic/item_index_id_dict.pkl', 'rb') as f:
            item_index_id_dict = pkl.load(f)
        with open('./Datasets/Electronic/item_id_index_dict.pkl', 'rb') as f:
            item_id_index_dict = pkl.load(f)
            
        with open('./Datasets/Electronic/category_item_dict.pkl', 'rb') as f:
            category_item_dict = pkl.load(f)
        with open('./Datasets/Electronic/item_categoory_dict.pkl', 'rb') as f:
            item_category_dict= pkl.load(f)
            
#         ipdb.set_trace()
        usernum = len(log_file_final.keys()) # 7176
        itemnum = len(item_index_id_dict.keys()) # 7506
        catenum = len(category_item_dict.keys()) # 407 
            
    user_num = 0
    interaction_num = 0
    for u,id_list in log_file_final.items():
        user_num += 1
        cate_list = log_category_final[u]
        for v in range(0,len(id_list)):
            interaction_num += 1
    ave_item_percate = 0
    ave_cate_peritem = 0
    item_percate_min = 100000000000
    item_percate_max = 0
    cate_peritem_min = 100000000000
    cate_peritem_max = 0
    for it, cate in item_category_dict.items():
        ave_cate_peritem += len(cate)
        if len(cate)<item_percate_min:
            item_percate_min = len(cate)
        if len(cate)>item_percate_max:
            item_percate_max = len(cate)
    for cate, it in category_item_dict.items():
        ave_item_percate += len(it)
        if len(it)<cate_peritem_min:
            cate_peritem_min = len(it)
        if len(it)>cate_peritem_max:
            cate_peritem_max = len(it)
    item_num = len(item_category_dict.keys())
    cate_num = len(category_item_dict.keys())
    ave_cate_peritem = ave_cate_peritem / item_num
    ave_item_percate = ave_item_percate / cate_num
    print("The number of user:", user_num) # 11378
    print("The number of item:", item_num) # 10580
    print("The number of cate:", cate_num) # 10580
    print("The number of interaction:", interaction_num) # 1004123
    print("The Average number of category of one item:", ave_cate_peritem) # 10580
    print("The min of category of one item:", cate_peritem_min) # 10580
    print("The max of category of one item:", cate_peritem_max) # 10580
    print("The Average number of item of one category:", ave_item_percate) # 10580
    print("The min of item of one category:", item_percate_min) # 10580
    print("The max of item of one category:", item_percate_max) # 10580


#         ipdb.set_trace()
    user_id_train = {}
    user_id_valid = {}
    user_id_test = {}
    user_time_train = {}
    user_time_valid = {}
    user_time_test = {}
    user_cate_train = {}
    user_cate_valid = {}
    user_cate_test = {}
    for u_index in log_file_final.keys():
        if len(log_file_final[u_index]) == len(log_timestep_final[u_index]) == len(log_category_final[u_index]):
            flag = 0
        else:
            flag = 1
        if flag == 1:
            print("Length wrong!!!")
            ipdb.set_trace()

        v_id = []
        v_time = []
        v_cate = []

        for item in log_file_final[u_index]:
            v_id.append(item)
        for item in log_timestep_final[u_index]:
            v_time.append(item)
        for item in log_category_final[u_index]:
            v_cate.append(item)

        user_id_train[u_index] = v_id[:-2]
        user_id_valid[u_index] = []
        user_id_valid[u_index].append(v_id[-2])
        user_id_test[u_index] = []
        user_id_test[u_index].append(v_id[-1])

        user_time_train[u_index] = v_time[:-2]
        user_time_valid[u_index] = []
        user_time_valid[u_index].append(v_time[-2])
        user_time_test[u_index] = []
        user_time_test[u_index].append(v_time[-1])

        user_cate_train[u_index] = v_cate[:-2]
        user_cate_valid[u_index] = []
        user_cate_valid[u_index].append(v_cate[-2])
        user_cate_test[u_index] = []
        user_cate_test[u_index].append(v_cate[-1])

    return [user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, item_category_dict]
   

    

# evaluate on test set
def evaluate_SeeDRec(model, dataset, args, user_list, diff_matrix, seq_id_fortest, sample_pool_fortest, pos_id_fortest):
    with torch.no_grad():
        print('Start test...')
        [user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, item_categoory_dict] = dataset
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        NDCG_20 = 0.0
        NDCG_50 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        HT_20 = 0.0
        HT_50 = 0.0
        AUC = 0.0
        user_tensor = np.array(user_list)
        labels = torch.zeros(args.num_samples, device='cuda')
        labels[0] = 1
        for i in range(0,int(user_tensor.shape[0]/args.batch_size)+1):
            u = user_tensor[i*args.batch_size: (i+1)*args.batch_size]
            seq_id = torch.tensor(seq_id_fortest[u],device='cuda')
            sample_pool = torch.tensor(sample_pool_fortest[u],device='cuda')
            pos_id = torch.tensor(pos_id_fortest[u],device='cuda')  
            random_indices = torch.randint(low=0, high=sample_pool_fortest[u].shape[1], size=[seq_id.shape[0],args.num_samples], device='cuda') # torch.Size([1015, 10])
            item_idx = torch.gather(input=sample_pool,dim=-1,index=random_indices) # torch.Size([128, 200, 2])
            item_idx[:,0] = pos_id.squeeze()
            diff_feat = diff_matrix[u]
            predictions = model.predict(u, seq_id, item_idx, diff_feat)
#             ipdb.set_trace()
            for j in range(0, predictions.shape[0]):
                AUC += roc_auc_score(labels.cpu(), predictions[j].cpu())  
                rank = (-predictions[j]).argsort().argsort()[0].item()
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if rank < 20:
                    NDCG_20 += 1 / np.log2(rank + 2)
                    HT_20 += 1
                if rank < 50:
                    NDCG_50 += 1 / np.log2(rank + 2)
                    HT_50 += 1
            print('process test batch {}'.format(i))
    return NDCG_1 / user_tensor.shape[0], NDCG_5 / user_tensor.shape[0], NDCG_10 / user_tensor.shape[0], NDCG_20 / user_tensor.shape[0], NDCG_50 / user_tensor.shape[0], HT_1 / user_tensor.shape[0], HT_5 / user_tensor.shape[0], HT_10 / user_tensor.shape[0], HT_20 / user_tensor.shape[0], HT_50 / user_tensor.shape[0], AUC / user_tensor.shape[0], 0.0