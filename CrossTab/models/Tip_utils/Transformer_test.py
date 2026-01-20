'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on Vision Transformer and BERT
* Based on AViT https://github.com/siyi-wind/AViT/blob/main/Models/Transformer/ViT_adapters.py
* Based on BLIP https://github.com/salesforce/BLIP/blob/main/models/med.py
'''
from typing import Dict, List
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange,repeat
import sys
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/multimodal/TIP')
from models.Tip_utils.pieces import DotDict
import torch.nn.functional as F
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, mask=None, visualize=False):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)
        # print(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        if visualize == False:
            return x
        else:
            return x, attn


class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.kv_proj = nn.Linear(k_dim,k_dim*2,bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim,k_dim)
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, q, k, visualize=False, mask=None):

        B,N_k,K = k.shape # (B, H)
        _,N_q,_ = q.shape
        kv = self.kv_proj(k).reshape(B,N_k,2,self.num_heads,K//self.num_heads).permute(2, 0, 3, 1, 4)  # 
        k,v = kv[0], kv[1]  # (B,H,N,C)
        q = self.q_proj(q).reshape(B,N_q,self.num_heads,K//self.num_heads).permute(0,2,1,3)  # (B,H,N,C)
        attn = (q @ k.transpose(-2,-1))*self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, K)
        out = self.proj(out)
        out = self.proj_drop(out)
        if visualize == False:
            return out
        else:
            return out, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, is_cross_attention=False, encoder_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = 0.5
        self.norm1 = norm_layer(dim)
        self.is_cross_attention = is_cross_attention
        self.attn = Attention(
        dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if self.is_cross_attention:
           self.cross_attn = CrossAttention(
               q_dim=dim, k_dim=encoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
           self.cross_norm = norm_layer(dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, encoder_hidden_states=None, mask=None, mask_cross=None, visualize=False):
        if visualize==False:
            # self attention
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            # cross attention
            if self.is_cross_attention:
                assert encoder_hidden_states is not None
                x = x + self.drop_path(self.cross_attn(self.cross_norm(x), encoder_hidden_states, mask=mask_cross))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        

# from .SwitchTransformer import MoE_Block
from .SwitchTransformer_proto import MoE_Block
import matplotlib.pyplot as plt

class TabularTransformerEncoder(nn.Module):
    '''
    Tabular Transformer Encoder based on BERT
    cat_lengths_tabular: categorical feature length list, e.g., [5,4,2]
    con_lengths_tabular: continuous feature length list, e.g., [1,1]
    '''

    def __init__(self, args: Dict) -> None:
        super(TabularTransformerEncoder, self).__init__()

        self.con_proj = nn.Linear(1, args.tabular_embedding_dim)
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.tabular_embedding_dim))
        self.mask_special_token = nn.Parameter(torch.zeros(1, 1, args.tabular_embedding_dim))

        # self.column_embedding = nn.Linear(768, args.tabular_embedding_dim)
        self.column_embedding = nn.Linear(768, 1)

        self.norm = nn.LayerNorm(args.tabular_embedding_dim)
        self.dropout = nn.Dropout(args.embedding_dropout) if args.embedding_dropout > 0. else nn.Identity()
        self.expert_centroids = nn.Parameter(
            torch.randn(5, args.tabular_embedding_dim)  # [num_routed_experts, d_model]
        )
        nn.init.xavier_uniform_(self.expert_centroids)
        print(self.expert_centroids.shape)

        # transformer

        # self.transformer_blocks = nn.ModuleList([
        #                     Block(dim=args.tabular_embedding_dim, drop=args.drop_rate, is_cross_attention=False)
        #                     for i in range(args.tabular_transformer_num_layers)
        #                     ])
        # else:
        self.transformer_blocks = nn.ModuleList([
            MoE_Block(dim=args.tabular_embedding_dim, expert_centroids=self.expert_centroids,  drop=args.drop_rate, share_num=args.share_num, is_cross_attention=False)
            for i in range(args.tabular_transformer_num_layers)
        ])

        self.fc = nn.Linear(args.tabular_embedding_dim, 3)
        
        if args.checkpoint is None:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.mask_special_token, std=.02)
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def embedding(self, x, mask_special=None, header_embedding=None):
        
        # print('h.shape', header_embedding.shape)
        x = self.con_proj(x.unsqueeze(-1))
        if mask_special is not None:
            mask_special = mask_special.unsqueeze(-1)
            mask_special_tokens = self.mask_special_token.expand(x.shape[0], x.shape[1], -1)
            x = mask_special*mask_special_tokens + (~mask_special)*x
        # concat
        if header_embedding is not None:
            column_embed = self.column_embedding(header_embedding)
            x = x * column_embed

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.norm(x)
        x = self.dropout(x)

        return x
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, mask_special: torch.Tensor=None, header_embedding: torch.Tensor=None) -> torch.Tensor:
        
        x = self.embedding(x, mask_special=mask_special, header_embedding=header_embedding)

        if mask is not None:
            B, N = mask.shape
            cls_mask = torch.zeros(B, 1).bool().to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask[:,None,:].repeat(1, N+1, 1)
            mask_eye = ~torch.eye(N+1).bool().to(mask.device)
            mask_eye = mask_eye[None,:,:]
            mask = mask * mask_eye
            mask = mask[:,None,:,:]
            mask = mask*(-1e9)
            assert x.shape[1] == mask.shape[2]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask)
            # x = transformer_block(x)

        x = self.fc(x[:,0,:])  # take the cls token output
        return x


if __name__ == "__main__":

    ### Attention Test
    # x = torch.randn(2, 5, 512)
    # mask = torch.tensor([[True, False, True, False, False],[False, True, False, False, False]])
    # B, N = mask.shape
    # mask = mask[:,None,:].repeat(1, N, 1)
    # mask_eye = ~torch.eye(N).bool()
    # mask_eye = mask_eye[None,:,:]
    # mask = mask * mask_eye
    # mask = mask[:,None,:,:]
    # mask = mask*(-1e9)
    # model = Attention(dim=512, num_heads=8)
    # out = model(x, mask=mask)
    # print(out.shape)

    # #### Cross Attention Test
    # x = torch.randn(2, 10, 512)
    # encoder_hidden_states = torch.randn(2, 12, 512)
    # model = Block(dim=512, num_heads=8, is_cross_attention=True, encoder_dim=512)
    # out = model(x, encoder_hidden_states)
    # print(out.shape)

    ### Tabular Transformer Encoder Test
    from torchvision.transforms import transforms
    import torchio as tio
    from datasets.ContrastiveReconstructImagingAndTabularHeaderDataset import ContrastiveReconstructImagingAndTabularHeaderDataset
    dataset = ContrastiveReconstructImagingAndTabularHeaderDataset(
        data_path_imaging='/mnt/data2/yibing/image-tabular/BrainMRI-CrossTab-Pretraining/data/data/Merge/train_path.pt',
        delete_segmentation=False,
        augmentation=transforms.Compose([
            tio.Resize((128, 128, 128)),  # 将图像调整为指定尺寸
            tio.Lambda(lambda x: x.float())  # 转换为 float 类型
        ]), augmentation_rate=0.5,
        data_path_tabular='/mnt/data2/yibing/image-tabular/BrainMRI-CrossTab-Pretraining/data/data/Merge/train.pt',
        corruption_rate=0.15, replace_random_rate=0.0, replace_special_rate=0.50,
        field_lengths_tabular='/mnt/data2/yibing/image-tabular/BrainMRI-CrossTab-Pretraining/data/data/Merge/tabular_length.pt',
        one_hot_tabular=False,
        labels_path='/mnt/data2/yibing/image-tabular/BrainMRI-CrossTab-Pretraining/data/data/Merge/labels_train.pt',
        img_size=[128, 128, 128], live_loading=True, augmentation_speedup=False
    )
    #
    import json
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        dataset,
        num_workers=4, batch_size=4,
        pin_memory=True, shuffle=True, persistent_workers=True, drop_last=True)
    first_batch = next(iter(val_loader))

    print(first_batch[-1].shape)

    # x = x.unsqueeze(0)
    # cat_lengths_tabular, con_lengths_tabular = [2, 4, 3], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # args = DotDict({'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'embedding_dropout': 0.1,
    #                 'embedding_dim': 2048, 'drop_rate': 0.0,
    #                 'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4})
    # print(cat_lengths_tabular, con_lengths_tabular)
    # print(torch.load('/mnt/iMVR/yibing/image-tabular/BrainMRI-CrossTab-Pretraining/data/data/adni_tabular/tabular_length_rev.pt'))
    # model = TabularTransformerEncoder(args, cat_lengths_tabular, con_lengths_tabular)
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #
    # out = model(x, mask=None, mask_special=None)
    # print(out.shape)
    #
    # multimodal_features = torch.randn(2, 19, 512)
    # model = TabularPredictor(args, cat_lengths_tabular, con_lengths_tabular)
    # out = model(multimodal_features)
    # for y in out:
    #     print(y.shape)


