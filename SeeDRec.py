import os
import time
import torch
import argparse
import ipdb
import io
import math
import random
from copy import deepcopy
from model import SeeDRec, EarlyStopping
from utils import *
import models.gaussian_diffusion as gd
from models.DNN import DNN
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# -*- coding: UTF-8 -*-
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=2000)

from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)


def get_diff(args, diffusion, user_cate_train, user_time_train, user_cate_valid, user_time_valid):
    print("Start inference via pre-trained diffusion model!")
    diff_matrix = torch.zeros([args.usernum+1, args.catenum+1], device='cuda')
    diff_matrix_test = torch.zeros([args.usernum+1, args.catenum+1], device='cuda')
    with torch.no_grad():
        t_start = time.time()
        for u in range(1, args.usernum + 1):
            if u not in user_list:
                continue
            # init the tensor  
            corpus_target_temp = np.zeros([args.catenum+1], dtype=np.float32)
            corpus_target_temp_test = np.zeros([args.catenum+1], dtype=np.float32)

            diff_matrix[u] = diffusion.generate_sample(args, model_diff, user_cate_train[u], user_time_train[u]).detach().squeeze()
            diff_matrix_test[u] = diffusion.generate_sample(args, model_diff, user_cate_train[u]+user_cate_valid[u], user_time_train[u]+user_time_valid[u]).detach().squeeze()
        t_end = time.time()
        print("Time interval of one epoch:{:.4f}".format(t_end-t_start))
    return diff_matrix, diff_matrix_test


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='str')
parser.add_argument('--cross_dataset', required=True)
# parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--num_samples', default=1000, type=int)
parser.add_argument('--decay', default=4, type=int)
parser.add_argument('--lr_decay_rate', default=0.99, type=float)
parser.add_argument('--index', default=0, type=int)
# parser.add_argument('--version', default=None, type=str)
# parser.add_argument('--rec_version', default='id', type=str)
parser.add_argument('--lr_linear', default=0.01, type=float)
parser.add_argument('--start_decay_linear', default=8, type=int)
parser.add_argument('--temperature', default=5, type=float)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)
parser.add_argument('--patience', default=5, type=int)
# parser.add_argument('--is_norm', default='True', type=str)

parser.add_argument('--lr_diff', type=float, default=0.00005, help='learning rate')
parser.add_argument('--weight_decay_diff', type=float, default=0.0)
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0005, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.005, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--reweight_version', type=str, default='AllLinear', help='in AllOne, AllLinear, MinMax')
parser.add_argument('--result_path', type=str, default=True, help='the path of result')
parser.add_argument('--filter_prob', type=float, default=0.1, help='the path of result')
parser.add_argument('--scale_weight', type=float, default=1.0, help='the path of result')
parser.add_argument('--is_minmaxnorm', type=int, default=1, help='the path of result')
parser.add_argument('--train_interval', type=int, default=20)


args = parser.parse_args()


SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

result_path = './results_mk/' + str(args.cross_dataset) + '/SeeDRec_SASRec/' + '/' + str(args.index) + 'th_seed' + str(args.seed) + '/'
save_date_path = './data_file/'+str(args.cross_dataset)+'/'
print("Save in path:", result_path)
if not os.path.isdir(result_path):
    os.makedirs(result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()
args.result_path = result_path
        
if __name__ == '__main__':
    dataset = data_partition(args.dataset, args.cross_dataset, args.maxlen)
    [user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, item_categoory_dict] = dataset
    print("The number of user:", usernum)
    print("The number of item:", itemnum)
    print("The number of category:", catenum)
    args.usernum = usernum
    args.itemnum = itemnum
    args.catenum = catenum
    interaction_num_all = 0.0
    for u_i in range(1, usernum+1):
        if len(user_id_train[u_i]) > 0:
            interaction_num_all = interaction_num_all + len(user_id_train[u_i])
    print('average sequence length is: %.2f' % (interaction_num_all / len(user_id_train.keys())))
    
    random_min = 1
    random_max = itemnum + 1
    item_entries = np.arange(start=1, stop=itemnum + 1, step=1, dtype=int)
    cate_entries = np.arange(start=1, stop=catenum+1, step=1, dtype=int)
    print("The min_ID is {} and the max_ID is {}".format(item_entries[0], item_entries[-1])) 
    print("The min_Cate is {} and the max_Cate is {}".format(cate_entries[0], cate_entries[-1])) 
        
    user_list = []
    for u_i in range(1, usernum+1):
        if len(user_id_train[u_i]) >= 2: 
            user_list.append(u_i)    
    num_batch = math.ceil(len(user_list) / args.batch_size) # 908
    seq_id_fortrain, pos_id_fortrain, seq_id_fortest, sample_pool_fortest, pos_id_fortest = generate_file(args, user_list, user_id_train, user_id_valid, user_id_test, item_entries, itemnum)
        
    model = SeeDRec(usernum, itemnum, catenum, args).cuda() # no ReLU activation in original SASRec implementation?
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
        
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location='cpu'))
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
        
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # set the early stop
    early_stopping = EarlyStopping(args.patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    # set the learning rate scheduler
    if args.lrscheduler == 'Steplr': # 
        learningrate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=args.decay, gamma=args.lr_decay_rate, verbose=True)
    elif args.lrscheduler == 'ExponentialLR': # 
        learningrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=args.lr_decay_rate, last_epoch=-1, verbose=True)
    elif args.lrscheduler == 'CosineAnnealingLR':
        learningrate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1, verbose=True)
        
    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    # ipdb.set_trace()
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, 'cuda').cuda()

    ### Build MLP ###
    out_dims = eval(args.dims) + [catenum+1] # [1000, 94949]
    in_dims = out_dims[::-1] # [94949, 1000]
    model_diff = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).cuda()

    optimizer_diff = torch.optim.AdamW(model_diff.parameters(), lr=args.lr_diff, weight_decay=args.weight_decay_diff)
    print("model_diff ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model_diff.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)

    if args.cross_dataset == 'Home':
        model_diff_path = './Checkpoints/DM_Home.pt'
        args.lr_diff=5e-5
        args.weight_decay_diff=0.1
        args.dims='[1000]'
        args.emb_size=10
        args.mean_type='x0'
        args.steps=10
        args.noise_scale=0.01
        args.noise_min=0.0005
        args.noise_max=0.005
        args.sampling_steps=0
        args.reweight=1
        args.w_min=0.1
        args.w_max=0.5
        args.log_name='log'
        args.reweight_version='AllOne'
        args.overlap_version='Sum'
        args.round=1
    elif args.cross_dataset == 'Electronic':
        model_diff_path = './Checkpoints/DM_Electronic.pt'
        args.lr_diff=5e-5
        args.weight_decay_diff=0.1
        args.dims='[1000]'
        args.emb_size=10
        args.mean_type='x0'
        args.steps=10
        args.noise_scale=0.1
        args.noise_min=0.0005
        args.noise_max=0.005
        args.sampling_steps=0
        args.reweight=1
        args.w_min=0.1
        args.w_max=0.5
        args.log_name='log'
        args.reweight_version='AllOne'
        args.overlap_version='Sum'
        args.round=1
        
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, 'cuda').cuda()
    model_diff = torch.load(model_diff_path).to('cuda')
    model_diff.eval()
    sasrec_model_path = './Checkpoints/checkpoint_'+args.cross_dataset+'.pt'
    get_updateModel(model, sasrec_model_path)
    sampler = WarpSampler(random_min, random_max, user_id_train, user_id_valid, user_id_test, user_time_train, user_time_valid, user_time_test, user_cate_train, user_cate_valid, user_cate_test, usernum, itemnum, catenum, user_list, args, item_categoory_dict, seq_id_fortrain, pos_id_fortrain, w_min = args.w_min, w_max = args.w_max, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, sample_ratio=1)
    
    diff_matrix, diff_matrix_test = get_diff(args, diffusion, user_cate_train, user_time_train, user_cate_valid, user_time_valid)
    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        t1 = time.time()
        loss_epoch = 0
        model.train()
        u_all_tensor = torch.tensor([], dtype=torch.int32)
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            user, seq_id, pos_id, neg_id = sampler.next_batch()   
            user, seq_id, pos_id, neg_id = np.array(user), np.array(seq_id), np.array(pos_id), np.array(neg_id)
            user, seq_id, pos_id, neg_id = torch.tensor(user,device='cuda'), torch.tensor(seq_id,device='cuda'), torch.tensor(pos_id,device='cuda'), torch.tensor(neg_id,device='cuda')
#                 ipdb.set_trace()
            user_pre_dist = torch.index_select(input=diff_matrix, dim=0, index=user)
            pos_logits, neg_logits = model(user, seq_id, pos_id, neg_id.squeeze(), user_pre_dist)
            indices = torch.where(pos_id != 0)
            pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
            
            adam_optimizer.zero_grad()
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            loss_epoch += loss.item()
#             for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
#             ipdb.set_trace()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) 
            with io.open(result_path + 'loss_log.txt', 'a', encoding='utf-8') as file:
                file.write("loss in epoch {} iteration {}: {}\n".format(epoch, step, loss.item()))
#         ipdb.set_trace()
        learningrate_scheduler.step()

        t2 = time.time()
        print("loss in epoch {}: {}, time: {}\n".format(epoch, loss_epoch / num_batch, t2 - t1)) 
        with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write("loss in epoch {}: {}, time: {}\n".format(epoch, loss_epoch / num_batch, t2 - t1))
        if epoch % args.train_interval == 0:
            model.eval()
            print('Evaluating', end='')
            t_test = evaluate_SeeDRec(model, dataset, args, user_list, diff_matrix_test, seq_id_fortest, sample_pool_fortest, pos_id_fortest)
            t3 = time.time()
            print('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:\n' % (epoch, t3-t1, t3-t0))
            print('        test: NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f\n' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11]))
            with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
                file.write('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:\n' % (epoch, t3-t1, t3-t0))
                file.write('        NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f\n' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11]))

    #         ipdb.set_trace()
            early_stopping(epoch, model, result_path, t_test)
    #         if epoch >= 180:
            if early_stopping.early_stop:
                print("Save in path:", result_path)
                print("Early stopping in the epoch {}, NDCG@1: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@20: {:.4f}, NDCG@50: {:.4f}, HR@1: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@20: {:.4f}, HR@50: {:.4f}, AUC: {:.4f}".format(early_stopping.save_epoch, early_stopping.best_performance[0], early_stopping.best_performance[1], early_stopping.best_performance[2], early_stopping.best_performance[3], early_stopping.best_performance[4], early_stopping.best_performance[5], early_stopping.best_performance[6], early_stopping.best_performance[7], early_stopping.best_performance[8], early_stopping.best_performance[9], early_stopping.best_performance[10]))
                with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
                    file.write("Early stopping in the epoch {}, NDCG@1: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@20: {:.4f}, NDCG@50: {:.4f}, HR@1: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@20: {:.4f}, HR@50: {:.4f}, AUC: {:.4f}\n".format(early_stopping.save_epoch, early_stopping.best_performance[0], early_stopping.best_performance[1], early_stopping.best_performance[2], early_stopping.best_performance[3], early_stopping.best_performance[4], early_stopping.best_performance[5], early_stopping.best_performance[6], early_stopping.best_performance[7], early_stopping.best_performance[8], early_stopping.best_performance[9], early_stopping.best_performance[10]))
                break

    sampler.close()
    
    
    
    
    
    
    
    
    
