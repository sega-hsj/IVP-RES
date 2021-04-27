import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Lang_Att_Head(nn.Module):
    def __init__(self,cfg,text_dim=1024):
        super().__init__()
        self.text_dim = text_dim
        self.out_dim = 2
        self.text_att = torch.nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.GroupNorm(32, self.text_dim),
            nn.ReLU(),
            nn.Linear(self.text_dim, self.out_dim),
        )

    def forward(self,word_embed,word_masks):
        B,L=word_embed.shape[:2]
        att_weight=self.text_att(word_embed.reshape(B*L,-1)).reshape(B,L,self.out_dim)
        att_weight=torch.softmax(att_weight,dim=-1)
        
        word_masks=word_masks.int().unsqueeze(-1).repeat(1,1,self.out_dim)
        att_weight=att_weight*word_masks

        A_att=att_weight[...,0].unsqueeze(1) #B,1,L
        A_lang=torch.matmul(A_att,word_embed).squeeze(1)
        A_lang=F.normalize(A_lang,p=2,dim=-1)
        return A_lang


class Lang_Feat_Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.EMB_PATH         != None

        self.emb_path               = cfg.EMB_PATH
        self.use_att                = cfg.USE_LANG_ATT
        self.text_dim               = 1024
        self.embed                  = np.load(self.emb_path)
        self.embed                  = torch.Tensor(self.embed).cuda()
        self.embed                  = torch.autograd.Variable(self.embed,requires_grad=True)
        self.gru                    = nn.GRU(input_size=300, hidden_size=self.text_dim//2, batch_first=True, bidirectional=True)
        self.att_head               = None
        if self.use_att:
            self.att_head           = Lang_Att_Head(cfg)
            
    def forward(self,batched_inputs):
        text_feats=[]
        word_masks=[]
        for x in batched_inputs:
            text_idx = x['text'].long().cuda()
            emb_feat = self.embed[text_idx]
            word_mask = (text_idx!=0)
            word_masks.append(word_mask)
            begin = 20 - word_mask.sum()
            text_feat=torch.zeros(20,self.text_dim).cuda()
            valid_text_feat=self.gru(emb_feat[begin:].unsqueeze(0))[0].squeeze()
            valid_text_feat=F.normalize(valid_text_feat,p=2,dim=-1)
            text_feat[begin:]=valid_text_feat
            text_feats.append(text_feat)
        text_feats=torch.stack(text_feats)#B,20,C
        word_masks=torch.stack(word_masks).cuda()
        if self.use_att:
            text_feat=self.att_head(text_feats,word_masks)
        else:
            text_feat=text_feats.mean(dim=1) #TODO check
        return text_feat