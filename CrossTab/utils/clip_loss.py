from typing import Tuple, List

import torch
from torch import nn
from einops import rearrange

class CLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               lambda_0: float = 0.5) -> None:
    super(CLIPLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)

    #logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
    logits = torch.matmul(out0, out1.T) / self.temperature
    labels = torch.arange(len(out0), device=out0.device)
    
    loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
    loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
    loss = loss_0 + loss_1
  
    return loss, logits, labels
  

import torch.nn.functional as F

class TokenCLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, temperature: float, lambda_0: float = 0.5) -> None:
    super(TokenCLIPLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    # out0: image embeddding, out1: tabular embedding
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)
    tab_emb_q = out1
    patch_emb_q = out0
    atten_sim = torch.bmm(tab_emb_q, patch_emb_q.permute(0, 2, 1))
    tab_num = tab_emb_q.size(1)
    bz = tab_emb_q.size(0)
    atten_scores = F.softmax(atten_sim / self.temperature, dim=-1)  # bz, 196, 111
    tab_atten_output = torch.bmm(atten_scores, patch_emb_q)
    tab_atten_output = F.normalize(tab_atten_output, dim=-1)
    # print(tab_atten_output.shape)
    tab_sim = torch.bmm(tab_emb_q, tab_atten_output.permute(0, 2, 1)) / self.temperature
    tab_sim_1 = rearrange(tab_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(tab_num).type_as(tab_emb_q).long().repeat(bz)
    loss_tab_1 = F.cross_entropy(tab_sim_1, targets)
    tab_sim_2 = rearrange(tab_sim, "b n1 n2 -> (b n2) n1")
    loss_tab_2 = F.cross_entropy(tab_sim_2, targets)

    loss = (loss_tab_1 + loss_tab_2) / 2.

    return loss


if __name__ == "__main__":

    loss = TokenCLIPLoss(temperature=0.1, lambda_0=0.5)  # torch.Size([12, 128]) torch.Size([12, 128]) torch.Size([12, 2048, 10, 7, 6]) torch.Size([12, 22, 512])
    x = torch.randn(4, 20, 128)
    y = torch.randn(4, 10, 128)
    print(loss(x, y))