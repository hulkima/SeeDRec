import numpy as np
import torch
import ipdb
import torch.nn.functional as F
from torch import Tensor
import math
import os
import io
import copy
import time
import random
import copy
from torch.autograd import Variable

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_performance = None
        self.early_stop = False
        self.ndcg_max = None
        self.save_epoch = None
        self.delta = delta

    def __call__(self, epoch, model, result_path, t_test):

        if self.ndcg_max is None:
            self.ndcg_max = t_test[2]
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
        elif t_test[2] < self.ndcg_max:
            self.counter += 1
            print(f'In the epoch: {epoch}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
            self.counter = 0

    def save_checkpoint(self, epoch, model, result_path, t_test):
        print(f'Validation loss in {epoch} decreased {self.ndcg_max:.4f} --> {t_test[2]:.4f}.  Saving model ...\n')
        with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
            file.write("NDCG@10 in epoch {} decreased {:.4f} --> {:.4f}, the HR@10 is {:.4f}, the AUC is {:.4f}, the loss_rec is {:.4f}. Saving model...\n".format(epoch, self.ndcg_max, t_test[2], t_test[7], t_test[10], t_test[11]))
        torch.save(model.state_dict(), os.path.join(result_path, 'checkpoint.pt')) 
        self.ndcg_max = t_test[2]
        
        
def get_nonzero_position(log):
    tmp = (log != 0).float()
    tmp2= tmp * torch.arange(tmp.shape[1], 0, -1, device='cuda')
    indices = torch.argmax(tmp2, 1, keepdim=True)
    indices = torch.arange(3, device='cuda:0') + torch.clamp(indices, min=3) - 3    
    log = log.scatter(1, indices, 999999)

    return indices, log
    
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    
    
    
class SASRec_Embedding_prompt(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_Embedding_prompt, self).__init__()

        self.item_num = item_num # 3416
        self.dev = args.device #'cuda'
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) #Embedding(3417, 50, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE Embedding(200, 50)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) #Dropout(p=0.2)

        self.attention_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.attention_layers = torch.nn.ModuleList() # 2 layers of MultiheadAttention
        self.forward_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.forward_layers = torch.nn.ModuleList() # 2 layers of PointWiseFeedForward

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) #LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate, batch_first=True) # MultiheadAttention((out_proj): NonDynamicallyQuantizableLinear(in_features=50, out_features=50, bias=True))
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm((50,), eps=1e-08, elementwise_affine=True)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, prompt):
#         ipdb.set_trace()
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5 # torch.Size([128, 200, 64])
        seqs = self.emb_dropout(seqs) # torch.Size([128, 200, 64])
        positions = torch.tile(torch.arange(0,log_seqs.shape[1]), [log_seqs.shape[0],1]).cuda() # torch.Size([128, 200])
            # add the position embedding
        timeline_mask_clean = (log_seqs == 0)
        seqs *= ~timeline_mask_clean.unsqueeze(-1) # broadcast in last dim
        log_seqs_aug, log_seqs = get_nonzero_position(log_seqs)
        test = torch.zeros((seqs.shape), device='cuda').scatter(1, log_seqs_aug.unsqueeze(2).expand(-1, -1, 64), prompt)
        seqs = seqs + test
        seqs += self.pos_emb(positions) 
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality, 200
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device='cuda')) #(200,200)

        for i in range(len(self.attention_layers)):
#             seqs = torch.transpose(seqs, 0, 1) # torch.Size([200, 128, 50])
            Q = self.attention_layernorms[i](seqs) #torch.Size([128, 200, 50])
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask) # torch.Size([128, 200, 50])
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs # torch.Size([128, 200, 50])
#             seqs = torch.transpose(seqs, 0, 1) # torch.Size([128, 200, 50])

            seqs = self.forward_layernorms[i](seqs) # torch.Size([128, 200, 50])
            seqs = self.forward_layers[i](seqs) # torch.Size([128, 200, 50])
            seqs *=  ~timeline_mask.unsqueeze(-1) # torch.Size([128, 200, 50])

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, prompt): # for training      
        log_feats = self.log2feats(log_seqs, prompt) # torch.Size([128, 200, 50]) user_ids hasn't been used yet

        return log_feats # pos_pred, neg_pred    
    
    
class Projector_prompt(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector_prompt, self).__init__()

        self.input_size = input_size
        self.output_size = output_size            
        self.tanh = torch.nn.Tanh()
        self.layer1 = torch.nn.Linear(self.input_size, self.output_size)
        self.layer2 = torch.nn.Linear(self.output_size, self.output_size)

    def forward(self, inputs):
        outputs = self.layer2(self.tanh(self.layer1(inputs)))
        return outputs
    
class SeeDRec(torch.nn.Module):
    def __init__(self, user_num, item_num, cate_num, args):
        super(SeeDRec, self).__init__()
#         ipdb.set_trace()
        self.sasrec_embedding = SASRec_Embedding_prompt(item_num, args)
        self.hidden_units = args.hidden_units
        self.dev = args.device #'cuda'
        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.temperature = args.temperature
        self.fname = args.dataset
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        self.projector_prompt1_1 = Projector_prompt(cate_num+1, args.hidden_units)
        self.projector_prompt1_2 = Projector_prompt(cate_num+1, args.hidden_units)
        self.projector_prompt1_3 = Projector_prompt(cate_num+1, args.hidden_units)

        
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, user_pre_dist): # for training      
#         ipdb.set_trace()
        prompt = self.projector_prompt1_1(user_pre_dist).unsqueeze(1)
        prompt = torch.cat([prompt, self.projector_prompt1_2(user_pre_dist).unsqueeze(1)], dim=1)
        prompt = torch.cat([prompt, self.projector_prompt1_3(user_pre_dist).unsqueeze(1)], dim=1)
        target_log_feats = self.sasrec_embedding(log_seqs, prompt) # torch.Size([128, 200, 50]) 
        pos_embs = self.sasrec_embedding.item_emb(pos_seqs) # torch.Size([128, 200, 50])
        neg_embs = self.sasrec_embedding.item_emb(neg_seqs) # torch.Size([128, 200, 50])
        
        # get the l2 norm for the target domain recommendation
        target_log_feats_l2norm = torch.nn.functional.normalize(target_log_feats, p=2, dim=-1)
        pos_embs_l2norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
        neg_embs_l2norm = torch.nn.functional.normalize(neg_embs, p=2, dim=-1)
        pos_logits = (target_log_feats_l2norm * pos_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        neg_logits = (target_log_feats_l2norm * neg_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])

        pos_logits = pos_logits * self.temperature
        neg_logits = neg_logits * self.temperature
        
        return pos_logits, neg_logits # pos_pred, neg_pred
    

    def predict(self, user_ids, log_seqs, item_indices, user_pre_dist): # for inference
#         ipdb.set_trace()
        prompt = self.projector_prompt1_1(user_pre_dist).unsqueeze(1)
        prompt = torch.cat([prompt, self.projector_prompt1_2(user_pre_dist).unsqueeze(1)], dim=1)
        prompt = torch.cat([prompt, self.projector_prompt1_3(user_pre_dist).unsqueeze(1)], dim=1)
        target_log_feats = self.sasrec_embedding(log_seqs,prompt) # torch.Size([1, 200, 50]) 
        item_embs = self.sasrec_embedding.item_emb(item_indices)

        # get the l2 norm for the target domain recommendation
        final_feat_l2norm = torch.nn.functional.normalize(target_log_feats[:, -1, :], p=2, dim=-1)
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=-1)

        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) 
        logits = logits * self.temperature
            
        return logits # torch.Size([1, 100])