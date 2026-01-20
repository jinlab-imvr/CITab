import torch
import torch.nn.functional as F
from numpy.ma.core import masked
from torch import Tensor, nn
from .Transformer import *

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
            self,
            dim,
            num_experts: int,
            capacity_factor: float = 1.0,
            epsilon: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):

        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        return gate_scores

class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
            self,
            dim: int,
            mlp_hidden_dim: int,
            num_experts: int = 5,
            capacity_factor: float = 1.0,
            drop=0., act_layer=nn.GELU,
            share_num=0
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = mlp_hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.share_num = share_num

        self.selective_experts = nn.ModuleList(
            [
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
                for _ in range(num_experts)
            ]
        )
        print('number of experts: %d' % num_experts)
        self.expert_centroids = nn.Parameter(
            torch.randn(num_experts, dim)  # [num_routed_experts, d_model]
        )
        nn.init.xavier_uniform_(self.expert_centroids)

    def forward(self, x: Tensor):
        
        input = x.clone()
        
        self.similarities = torch.matmul(x, self.expert_centroids.transpose(0, 1)) #[batch, seq, N_r]
        gate_scores = F.softmax(self.similarities, dim = -1)  #[batch, seq, N_r]

        selective_expert_outputs = [expert(x) for expert in self.selective_experts]
        selective_stacked_expert_outputs = torch.stack(selective_expert_outputs, dim=-1)
        selective_moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * selective_stacked_expert_outputs, dim=-1
        )
        
        return selective_moe_output
       
class MoE_Block(nn.Module):
    def __init__(self, dim, num_heads=8, is_cross_attention=False, encoder_dim=None, share_num=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
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
        share_num = 5
        self.mlp = SwitchMoE(dim=dim, mlp_hidden_dim=int(dim * 1.75), act_layer=act_layer, drop=drop, num_experts=share_num)
        

    def forward(self, x, encoder_hidden_states=None, mask=None, visualize=False):
        if visualize==False:
            # self attention
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            # cross attention
            if self.is_cross_attention:
                assert encoder_hidden_states is not None
                x = x + self.drop_path(self.cross_attn(self.cross_norm(x), encoder_hidden_states))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            tmp, self_attn = self.attn(self.norm1(x), mask=mask, visualize=visualize)
            x = x+self.drop_path(tmp)
            if self.is_cross_attention:
                assert encoder_hidden_states is not None
                tmp, cross_attn = self.cross_attn(self.cross_norm(x), encoder_hidden_states, visualize=visualize)
                x = x+self.drop_path(tmp)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, {'self_attn':self_attn, 'cross_attn':cross_attn if self.is_cross_attention else None}

if __name__ == "__main__":

    x = torch.randn((4, 4, 256))
    net = MoE_Block(dim=256)
    print(net(x).shape)

